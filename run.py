import yaml
import os
import time
import traceback
import platform
import psutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import docker

from src.managers.llm_api.api_manager import LLMAPIManager
from src.managers.log.logger import Logger
from src.managers.image_builder.build_image import SWEBenchLoader

# Import for SWE-bench image builder testing
from src.managers.image_builder.build_image import SWEBenchImageBuilder
from swebench.harness.utils import load_swebench_dataset, EvaluationError
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.docker_build import (
    build_env_images, 
    build_instance_image, 
    setup_logger, 
    close_logger
)
from swebench.harness.docker_utils import (
    cleanup_container,
    copy_to_container,
    exec_run_with_timeout,
)
from swebench.harness.grading import get_eval_report, get_logs_eval
from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    DOCKER_PATCH,
    DOCKER_USER,
    DOCKER_WORKDIR,
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
    LOG_TEST_OUTPUT,
    UTF8,
)

# Git apply commands for patch application
GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]


def calculate_optimal_max_workers(ram_gb_per_worker: int = 6) -> int:
    """
    Calculate optimal max_workers based on system RAM.
    
    Args:
        ram_gb_per_worker: GB of RAM required per worker (default: 6GB)
        
    Returns:
        int: Optimal number of workers, minimum 1, maximum 8
    """
    try:
        # Get system memory in bytes
        memory = psutil.virtual_memory()
        total_ram_gb = memory.total / (1024**3)  # Convert bytes to GB
        
        # Calculate max workers based on RAM
        calculated_workers = int(total_ram_gb / ram_gb_per_worker)
        
        # Apply reasonable bounds: minimum 1, maximum 8
        optimal_workers = max(1, min(calculated_workers, 8))
        
        return optimal_workers
        
    except Exception as e:
        print(f"⚠️  Could not detect system RAM: {e}")
        print("🔧 Using default max_workers: 1")
        return 1

def format_test_results(tests_status: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper function to format test results for display.
    
    Args:
        tests_status: The tests_status dictionary from evaluation report
        
    Returns:
        Dictionary with formatted test results
    """
    f2p_success = len(tests_status.get("FAIL_TO_PASS", {}).get("success", []))
    f2p_failure = len(tests_status.get("FAIL_TO_PASS", {}).get("failure", []))
    p2p_success = len(tests_status.get("PASS_TO_PASS", {}).get("success", []))
    p2p_failure = len(tests_status.get("PASS_TO_PASS", {}).get("failure", []))
    
    return {
        "fail_to_pass": {
            "success": f2p_success,
            "failure": f2p_failure,
            "total": f2p_success + f2p_failure,
            "passed": f2p_failure == 0 and f2p_success > 0
        },
        "pass_to_pass": {
            "success": p2p_success,
            "failure": p2p_failure,
            "total": p2p_success + p2p_failure,
            "passed": p2p_failure == 0 and p2p_success > 0
        }
    }


def run_tests_on_container(
    container,
    test_spec,
    log_dir: Path,
    logger,
    timeout: int,
    test_prefix: str = ""
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]], str]:
    """
    Helper function to run tests on a container and return results.
    
    Returns:
        tuple: (test_results, evaluation_report, test_output)
    """
    # Write evaluation script to container
    eval_file = log_dir / f"{test_prefix}eval.sh"
    eval_file.write_text(test_spec.eval_script, encoding=UTF8)
    copy_to_container(container, eval_file, Path("/root/eval.sh"))
    
    # Prepare the run command
    run_command = "/bin/bash /root/eval.sh"
    
    # Execute the evaluation with timeout
    test_output, timed_out, exec_time = exec_run_with_timeout(
        container, run_command, timeout
    )
    
    # Save test output to file
    test_output_file = log_dir / f"{test_prefix}{LOG_TEST_OUTPUT}"
    test_output_file.write_text(test_output, encoding=UTF8)
    
    # Parse test results
    eval_status_map, found = get_logs_eval(test_spec, str(test_output_file))
    
    test_results = None
    if not found:
        test_results = {"status": "parse_failed", "output": test_output}
    else:
        test_results = eval_status_map
    
    # Generate evaluation report
    prediction = {
        KEY_INSTANCE_ID: test_spec.instance_id,
        KEY_PREDICTION: "",  # Empty prediction for pre-patch test
    }
    
    evaluation_report = get_eval_report(
        test_spec,
        prediction,
        str(test_output_file),
        include_tests_status=True,
    )
    
    return test_results, evaluation_report, test_output


def test_image(
    image_name: str,
    instance_id: str,
    dataset_name: str = "SWE-bench/SWE-bench_Lite",
    split: str = "test",
    timeout: int = 600,
    log_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tests a patch on an existing Docker image by running tests both before and after applying the gold patch.
    """
    start_time = time.time()
    result = {
        "instance_id": instance_id,
        "image_name": image_name,
        "patch_applied": False,
        "pre_patch_tests_passed": False,
        "post_patch_tests_passed": False,
        "pre_patch_test_results": None,
        "post_patch_test_results": None,
        "pre_patch_evaluation_report": None,
        "post_patch_evaluation_report": None,
        "pre_patch_terminal_output": None,
        "post_patch_terminal_output": None,
        "error": None,
        "logs": None,
    }
    
    try:
        # Load the specific instance from the dataset to get the gold patch
        dataset = load_swebench_dataset(dataset_name, split, [instance_id])
        
        if not dataset:
            raise ValueError(f"Instance {instance_id} not found in dataset {dataset_name}")
        
        instance = dataset[0]
        
        # Create test spec from the instance
        test_spec = make_test_spec(instance)
        
        # Set up Docker client
        client = docker.from_env()
        
        # Set up log directory
        if log_dir is None:
            log_dir = Path.cwd() / "logs" / "test_image" / instance_id
        else:
            log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        result["logs"] = str(log_dir)
        
        # Set up logger
        log_file = log_dir / "test_image.log"
        logger = setup_logger(instance_id, log_file)
        
        try:
            # Check if image exists
            try:
                client.images.get(image_name)
            except docker.errors.ImageNotFound:
                raise ValueError(f"Image {image_name} not found")
            
            # Create container from the image
            container_name = f"test_{instance_id}_{int(time.time())}"
            
            # Get run args and platform from test_spec
            run_args = test_spec.docker_specs.get("run_args", {})
            cap_add = run_args.get("cap_add", [])
            
            container = client.containers.create(
                image_name,
                name=container_name,
                working_dir=DOCKER_WORKDIR,
                user=DOCKER_USER,
                detach=True,
                command="tail -f /dev/null",  # Keep container running
                platform=test_spec.platform,  # Ensure correct architecture
                cap_add=cap_add,  # Add capabilities if needed
            )
            
            try:
                # Start the container
                container.start()
                
                # Run tests BEFORE applying the patch
                pre_patch_test_results, pre_patch_evaluation_report, pre_patch_output = run_tests_on_container(
                    container, test_spec, log_dir, logger, timeout, "pre_patch_"
                )
                
                result["pre_patch_test_results"] = pre_patch_test_results
                result["pre_patch_evaluation_report"] = pre_patch_evaluation_report
                result["pre_patch_terminal_output"] = pre_patch_output
                
                # Check if pre-patch tests passed and show results
                if pre_patch_evaluation_report:
                    instance_report = pre_patch_evaluation_report.get(instance_id, {})
                    result["pre_patch_tests_passed"] = instance_report.get("resolved", False)
                    
                    # Show detailed results
                    if "tests_status" in instance_report:
                        tests_status = instance_report["tests_status"]
                        formatted_results = format_test_results(tests_status)
                        result["pre_patch_test_details"] = formatted_results
                        
                        f2p = formatted_results["fail_to_pass"]
                        p2p = formatted_results["pass_to_pass"]
                        
                        print(f"Pre-patch test results:")
                        print(f"  FAIL_TO_PASS: {f2p['success']}/{f2p['total']} passed {'✅' if f2p['passed'] else '❌'}")
                        print(f"  PASS_TO_PASS: {p2p['success']}/{p2p['total']} passed {'✅' if p2p['passed'] else '❌'}")
                    else:
                        result["pre_patch_test_details"] = None
                        print("Pre-patch: Could not determine detailed test results")
                else:
                    result["pre_patch_test_details"] = None
                    print("Pre-patch: Could not determine test results")
                
                # Get the gold patch from the instance
                gold_patch = instance.get("patch", "")
                if not gold_patch:
                    raise ValueError(f"No gold patch found for instance {instance_id}")
                
                # Write patch to file and copy to container
                patch_file = log_dir / "gold_patch.diff"
                patch_file.write_text(gold_patch, encoding=UTF8)
                logger.info(f"Gold patch written to {patch_file}")
                
                copy_to_container(container, patch_file, Path(DOCKER_PATCH))
                logger.info(f"Gold patch copied to container at {DOCKER_PATCH}")
                
                # Apply the gold patch to the container
                print("Applying gold patch to container...")
                applied_patch = False
                for git_apply_cmd in GIT_APPLY_CMDS:
                    try:
                        exit_code, output = container.exec_run(
                            f"{git_apply_cmd} {DOCKER_PATCH}",
                            workdir=DOCKER_WORKDIR,
                            user=DOCKER_USER,
                        )
                        if exit_code == 0:
                            logger.info(f"{APPLY_PATCH_PASS}:\n{output.decode(UTF8)}")
                            applied_patch = True
                            break
                        else:
                            logger.info(f"Failed to apply patch with {git_apply_cmd}: {output.decode(UTF8)}")
                    except Exception as e:
                        logger.info(f"Error applying patch with {git_apply_cmd}: {str(e)}")
                        continue
                
                if not applied_patch:
                    error_msg = f"{APPLY_PATCH_FAIL}: Could not apply gold patch with any method"
                    logger.error(error_msg)
                    raise EvaluationError(instance_id, error_msg, logger)
                
                result["patch_applied"] = True
                print("✅ Gold patch applied successfully")
                
                # Get git diff to see what changed
                _, git_diff_bytes = container.exec_run(
                    "git -c core.fileMode=false diff", workdir=DOCKER_WORKDIR
                )
                git_diff_output = git_diff_bytes.decode(UTF8).strip()
                logger.info(f"Git diff after patch application:\n{git_diff_output}")
                
                # Run tests AFTER applying the patch
                print("Running tests after applying patch...")
                post_patch_test_results, post_patch_evaluation_report, post_patch_output = run_tests_on_container(
                    container, test_spec, log_dir, logger, timeout, "post_patch_"
                )
                
                result["post_patch_test_results"] = post_patch_test_results
                result["post_patch_evaluation_report"] = post_patch_evaluation_report
                result["post_patch_terminal_output"] = post_patch_output
                
                # Check if post-patch tests passed
                if post_patch_evaluation_report:
                    instance_report = post_patch_evaluation_report.get(instance_id, {})
                    result["post_patch_tests_passed"] = instance_report.get("resolved", False)
                    
                    # Show detailed results
                    if "tests_status" in instance_report:
                        tests_status = instance_report["tests_status"]
                        formatted_results = format_test_results(tests_status)
                        result["post_patch_test_details"] = formatted_results
                        
                        f2p = formatted_results["fail_to_pass"]
                        p2p = formatted_results["pass_to_pass"]
                        
                        print(f"Post-patch test results:")
                        print(f"  FAIL_TO_PASS: {f2p['success']}/{f2p['total']} passed {'✅' if f2p['passed'] else '❌'}")
                        print(f"  PASS_TO_PASS: {p2p['success']}/{p2p['total']} passed {'✅' if p2p['passed'] else '❌'}")
                        
                        if result["post_patch_tests_passed"]:
                            print("✅ All post-patch tests passed!")
                        else:
                            print("❌ Some post-patch tests failed")
                    else:
                        result["post_patch_test_details"] = None
                        print("Post-patch: Could not determine detailed test results")
                else:
                    result["post_patch_test_details"] = None
                    print("⚠️  Could not determine post-patch test results")
                
            finally:
                # Clean up container
                cleanup_container(client, container, logger)
                
        finally:
            close_logger(logger)
            
    except Exception as e:
        error_msg = f"Error testing image {image_name} with instance {instance_id}: {str(e)}"
        print(f"❌ {error_msg}")
        result["error"] = error_msg
        print(f"Traceback: {traceback.format_exc()}")
    
    return result


def test_swe_image_builder(
    instance_ids: List[str],
    dataset_name: str = "SWE-bench/SWE-bench_Lite",
    split: str = "test",
    max_workers: int = 2,
    force_rebuild: bool = False,
    timeout: int = 600,
) -> Dict[str, Any]:
    """
    Test the SWEBenchImageBuilder by building images for multiple instances and running evaluations.
    
    This function provides a comprehensive testing framework for the SWEBenchImageBuilder class.
    It performs the following operations:
    
    1. **Image Building**: Uses SWEBenchImageBuilder to build Docker images for specified SWE-bench instances
    2. **Pre-patch Testing**: Runs tests on the built images before applying any patches
    3. **Patch Application**: Applies the gold patch from the SWE-bench dataset to the container
    4. **Post-patch Testing**: Runs tests again after patch application to verify the fix
    5. **Result Analysis**: Compares pre-patch and post-patch test results to determine success
    
    Usage Examples:
    
    # Basic usage with default parameters
    results = test_swe_image_builder(["django__django-10914"])
    
    # Test multiple instances with custom settings
    results = test_swe_image_builder(
        instance_ids=["django__django-10914", "astropy__astropy-12907"],
        dataset_name="SWE-bench/SWE-bench",
        split="test",
        max_workers=1,
        force_rebuild=True,
        timeout=900
    )
    
    # Test with SWE-bench Lite dataset
    results = test_swe_image_builder(
        instance_ids=["django__django-10914"],
        dataset_name="SWE-bench/SWE-bench_Lite",
        split="dev"
    )
    
    Args:
        instance_ids (List[str]): List of SWE-bench instance IDs to build and test.
                                 Format: "repository__repository-issue_number"
                                 Example: ["django__django-10914", "astropy__astropy-12907"]
        
        dataset_name (str, optional): Name of the SWE-bench dataset to use.
                                    Defaults to "SWE-bench/SWE-bench_Lite".
                                    Options: "SWE-bench/SWE-bench", "SWE-bench/SWE-bench_Lite"
        
        split (str, optional): Dataset split to use. Defaults to "test".
                              Options: "dev", "test"
        
        max_workers (int, optional): Number of parallel workers for image building.
                                   Defaults to 2. Use 1 for sequential building to avoid memory issues.
        
        force_rebuild (bool, optional): Whether to force rebuild existing images.
                                      Defaults to False. Set to True to rebuild even if images exist.
        
        timeout (int, optional): Timeout in seconds for test execution. Defaults to 600 (10 minutes).
    
    Returns:
        Dict[str, Any]: Comprehensive test results containing:
            - instance_results: Dictionary mapping instance_id to detailed results
            - errors: List of error messages encountered during testing
            
        Each instance result contains:
            - image_name: Name of the built Docker image
            - build_status: Status of the image build process
            - test_result: Detailed test results including:
                - patch_applied: Whether the gold patch was successfully applied
                - pre_patch_tests_passed: Whether tests passed before patch application
                - post_patch_tests_passed: Whether tests passed after patch application
                - pre_patch_test_details: Detailed breakdown of pre-patch test results
                - post_patch_test_details: Detailed breakdown of post-patch test results
                - logs: Path to log files for debugging
    
    Test Result Interpretation:
        - ✅ PERFECT: Pre-patch tests failed, post-patch tests passed (patch fixed the issue)
        - ✅ GOOD: All tests pass both before and after patch (no regression)
        - ⚠️  WARNING: Pre-patch tests passed, post-patch tests failed (patch broke something)
        - ❌ ISSUE: Tests fail both before and after patch (patch didn't fix the issue)
    
    Requirements:
        - Docker must be installed and running
        - SWE-bench package must be installed
        - Sufficient disk space for Docker images (several GB per instance)
        - Network access to download datasets and base images
    
    Error Handling:
        - Gracefully handles Docker image build failures
        - Continues testing other instances if one fails
        - Provides detailed error messages and stack traces
        - Logs all operations for debugging purposes
    
    Performance Notes:
        - Image building can take 10-30 minutes per instance
        - Test execution typically takes 5-15 minutes per instance
        - Use max_workers=1 for memory-constrained environments
        - Consider using SWE-bench_Lite for faster testing with smaller instances
    """
    start_time = time.time()
    results = {
        "instance_results": {},
        "errors": []
    }
    
    try:
        print(f"🚀 Starting SWEBenchImageBuilder test with {len(instance_ids)} instances")
        print(f"Instance IDs: {instance_ids}")
        
        # Step 1: Build images using SWEBenchImageBuilder
        print("\n📦 Step 1: Building images with SWEBenchImageBuilder...")
        builder = SWEBenchImageBuilder(
            dataset_name=dataset_name,
            split=split,
            instance_ids=instance_ids,
            max_workers=max_workers,
            force_rebuild=force_rebuild,
        )
        
        # Step 2: Test each successfully built image
        print("\n🧪 Step 2: Testing built images...")
        
        for instance_id in instance_ids:
            print(f"\n--- Testing instance: {instance_id} ---")
            
            try:
                # Get image name from builder
                image_name = builder.get_image_name(instance_id)
                build_status = builder.get_build_status(instance_id)
                
                print(f"Image name: {image_name}")
                print(f"Build status: {build_status}")
                
                if build_status in ['successful', 'already_exists']:
                    # Test the image
                    test_result = test_image(
                        image_name=image_name,
                        instance_id=instance_id,
                        dataset_name=dataset_name,
                        split=split,
                        timeout=timeout,
                    )
                    
                    results["instance_results"][instance_id] = {
                        "image_name": image_name,
                        "build_status": build_status,
                        "test_result": test_result,
                        "success": test_result.get("patch_applied", False)
                    }
                    
                    if test_result.get("patch_applied", False):
                        print(f"✅ {instance_id}: Image built and patch applied successfully")
                        
                        # Show test results summary
                        pre_patch_passed = test_result.get('pre_patch_tests_passed', False)
                        post_patch_passed = test_result.get('post_patch_tests_passed', False)
                        pre_patch_details = test_result.get('pre_patch_test_details')
                        post_patch_details = test_result.get('post_patch_test_details')
                        
                        print(f"  Pre-patch tests: {'✅ PASSED' if pre_patch_passed else '❌ FAILED'}")
                        if pre_patch_details:
                            f2p = pre_patch_details["fail_to_pass"]
                            p2p = pre_patch_details["pass_to_pass"]
                            print(f"    FAIL_TO_PASS: {f2p['success']}/{f2p['total']} {'✅' if f2p['passed'] else '❌'}")
                            print(f"    PASS_TO_PASS: {p2p['success']}/{p2p['total']} {'✅' if p2p['passed'] else '❌'}")
                        
                        print(f"  Post-patch tests: {'✅ PASSED' if post_patch_passed else '❌ FAILED'}")
                        if post_patch_details:
                            f2p = post_patch_details["fail_to_pass"]
                            p2p = post_patch_details["pass_to_pass"]
                            print(f"    FAIL_TO_PASS: {f2p['success']}/{f2p['total']} {'✅' if f2p['passed'] else '❌'}")
                            print(f"    PASS_TO_PASS: {p2p['success']}/{p2p['total']} {'✅' if p2p['passed'] else '❌'}")
                        
                        if not pre_patch_passed and post_patch_passed:
                            print("  🎯 PERFECT: Patch fixed failing tests!")
                        elif pre_patch_passed and post_patch_passed:
                            print("  ✅ GOOD: All tests pass both before and after patch")
                        elif pre_patch_passed and not post_patch_passed:
                            print("  ⚠️  WARNING: Patch broke previously passing tests")
                        else:
                            print("  ❌ ISSUE: Tests fail both before and after patch")
                    else:
                        print(f"❌ {instance_id}: Patch could not be applied")
                        results["errors"].append(f"{instance_id}: Patch application failed")
                        
                else:
                    print(f"❌ {instance_id}: Image build failed (status: {build_status})")
                    results["instance_results"][instance_id] = {
                        "image_name": None,
                        "build_status": build_status,
                        "test_result": None,
                        "success": False
                    }
                    results["errors"].append(f"{instance_id}: Image build failed")
                    
            except Exception as e:
                error_msg = f"Error testing instance {instance_id}: {str(e)}"
                print(f"❌ {error_msg}")
                results["errors"].append(error_msg)
                results["instance_results"][instance_id] = {
                    "image_name": None,
                    "build_status": "error",
                    "test_result": None,
                    "success": False,
                    "error": error_msg
                }
        
    except Exception as e:
        error_msg = f"Error in test_swe_image_builder: {str(e)}"
        print(f"❌ {error_msg}")
        results["errors"].append(error_msg)
        print(f"Traceback: {traceback.format_exc()}")
    
    return results


def load_config(config_path: str = "src/config/config.yaml") -> dict:
    """加载 YAML 配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def load_swe_bench_data(config: dict, logger):
    """加载和处理 SWE-bench 数据集"""
    dataset_config = config.get("dataset", {})
    workspace_config = config.get("workspace", {})
    
    dataset_name = dataset_config.get("name", "princeton-nlp/SWE-bench_Lite")
    split_name = dataset_config.get("split", "dev")
    workspace_path = workspace_config.get("path", "workspace")
    
    logger.info(f"开始处理 SWE-bench 数据集: {dataset_name}")
    
    # 创建数据加载器
    swe_loader = SWEBenchLoader(
        dataset_name=dataset_name,
        split_name=split_name,
        workspace_path=workspace_path,
        logger=logger
    )
    
    # 加载并处理数据集（限制处理数量用于测试）
    max_items = int(os.getenv("MAX_ITEMS", "5"))  # 默认处理5个，可通过环境变量调整
    result = swe_loader.load_and_process_all(max_items=max_items)
    
    # 显示统计信息
    stats = swe_loader.get_stats()
    logger.info(f"数据处理统计: {stats}")
    
    print(f"\n📊 SWE-bench 数据处理完成:")
    print(f"   数据集: {dataset_name}")
    print(f"   分割: {split_name}")
    print(f"   工作空间: {workspace_path}")
    print(f"   处理结果: {result}")
    print(f"   统计信息: {stats}")
    
    return swe_loader, result

def test_embedding_functionality(config: dict, logger):
    """测试 embedding 功能"""
    rag_config = config.get("rag", {})
    embedding_config = rag_config.get("embedding", {})
    
    # 检查是否启用 embedding
    if not embedding_config.get("enabled", False):
        logger.info("Embedding 功能未启用，跳过测试")
        print("⏭️  Embedding 功能未启用，跳过测试")
        return
    
    client_name = embedding_config.get("client", "openai")
    model_name = embedding_config.get("model", "text-embedding-3-small")
    
    logger.info(f"开始测试 Embedding 功能 - 客户端: {client_name}, 模型: {model_name}")
    
    print("🧠 RAG Embedding 功能测试")
    print("=" * 60)
    print(f"🔧 配置: 客户端={client_name}, 模型={model_name}")
    print("=" * 60)
    
    # 测试文本
    test_texts = [
        "人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。",
        "Machine learning algorithms can automatically learn and improve from experience.",
        "深度学习使用多层神经网络来处理和分析复杂的数据模式。",
        "自然语言处理帮助计算机理解和生成人类语言。"
    ]
    
    print(f"📝 测试文本 ({len(test_texts)} 条):")
    for i, text in enumerate(test_texts, 1):
        print(f"   {i}. {text}")
        logger.debug(f"测试文本 {i}: {text}")
    
    try:
        # 使用 LLMAPIManager 创建统一的客户端管理器
        manager = LLMAPIManager(
            client_name=client_name,
            timeout=30,
            max_retries=2,
            logger=logger
        )
        
        logger.info(f"成功创建 {client_name.upper()} 客户端管理器")
        print(f"✅ 成功创建 {client_name.upper()} 客户端管理器")
        
        print(f"\n🚀 开始生成嵌入向量...")
        logger.info(f"开始调用 embedding API - 模型: {model_name}")
        
        # 通过 LLMAPIManager 调用 embedding API（支持批量处理）
        response = manager.create_embeddings(
            input_text=test_texts,  # 直接传递文本列表，支持批量处理
            model=model_name,
            timeout=30,
            retry=2
        )
        
        # 检查响应是否成功
        if response is None:
            error_msg = "Embedding 生成失败: 所有重试都失败"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            return
        
        # 记录成功结果
        logger.info(f"Embedding 生成成功 - 模型: {response.model}, 向量数量: {len(response.data)}, Token使用: {response.usage.total_tokens}")
        
        print("✅ 嵌入向量生成成功!")
        print(f"\n📊 响应统计:")
        print(f"   🤖 使用模型: {response.model}")
        print(f"   📈 嵌入向量数量: {len(response.data)}")
        print(f"   🔢 Token 使用: {response.usage.prompt_tokens} prompt + {response.usage.total_tokens} total")
        
        # 显示每个嵌入向量的详细信息
        print(f"\n🔍 嵌入向量详情:")
        total_dimensions = 0
        for i, embedding_data in enumerate(response.data):
            vector_dim = len(embedding_data.embedding)
            total_dimensions += vector_dim
            first_few = embedding_data.embedding[:3]  # 显示前3个值
            last_few = embedding_data.embedding[-3:]  # 显示后3个值
            
            print(f"   向量 {i+1}: 维度={vector_dim}")
            print(f"           前3个值: {[round(x, 6) for x in first_few]}")
            print(f"           后3个值: {[round(x, 6) for x in last_few]}")
            
            # 记录到日志
            logger.debug(f"向量 {i+1}: 维度={vector_dim}, 索引={embedding_data.index}")
        
        avg_dimension = total_dimensions // len(response.data) if response.data else 0
        logger.info(f"平均向量维度: {avg_dimension}")
        
        print(f"\n🎯 测试总结:")
        print(f"   ✅ 成功生成 {len(response.data)} 个嵌入向量")
        print(f"   📏 平均向量维度: {avg_dimension}")
        print(f"   ⚡ Token 效率: {response.usage.total_tokens / len(test_texts):.1f} tokens/text")
        print(f"   🎉 Embedding 功能测试完成!")
        
        logger.info("Embedding 功能测试成功完成")
        
        # 关闭管理器
        manager.close()
        
    except Exception as e:
        error_msg = f"Embedding 测试失败: {str(e)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        
        # 提供故障排除建议
        print(f"\n💡 故障排除建议:")
        print(f"   1. 检查 API 密钥是否正确设置")
        print(f"   2. 确认网络连接正常")
        print(f"   3. 验证模型名称是否正确: {model_name}")
        print(f"   4. 检查 API 配额是否充足")

def test_provider_models(config: dict, logger):
    """测试所有配置的提供商和模型"""
    providers = config.get("providers", {})
    
    # 直接在代码中定义测试参数
    test_message = "四大名著有哪些？请简要介绍每一部。"
    system_message = "你是一个有用的AI助手，请用简洁明了的方式回答问题。"
    temperature = 0.1
    max_tokens = 500
    stream = False
    timeout = 30
    
    print("🚀 LLM API 多提供商模型测试")
    print("=" * 80)
    print(f"📝 测试消息: {test_message}")
    print(f"🔧 配置: stream={stream}, temperature={temperature}, max_tokens={max_tokens}")
    print("=" * 80)
    
    total_tests = 0
    successful_tests = 0
    
    # 遍历每个提供商
    for provider_name, models in providers.items():
        logger.info(f"开始测试提供商: {provider_name}")
        print(f"\n🏢 测试提供商: {provider_name.upper()}")
        print("-" * 60)
        
        try:
            # 创建该提供商的管理器
            manager = LLMAPIManager(
                client_name=provider_name,
                stream=stream,
                timeout=timeout,
                logger=logger
            )
            
            logger.info(f"{provider_name} 客户端创建成功")
            print(f"✅ {provider_name} 客户端创建成功")
            
            if provider_name == "private":
                model_name = os.getenv("PRIVATE_MODEL_NAME", "qwen-2.5-coder-3b-instruct")
                total_tests += 1
                response = manager.chat(
                    model=model_name,
                    message=test_message,
                    system_message=system_message,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                if response is not None:
                    logger.info(f"模型 {model_name} 测试成功，响应长度: {len(response)} 字符")
                    print(f"📤 请求成功")
                    print(f"📥 响应内容:")
                    print(f"   {response}")
                    print(f"✅ 模型 {model_name} 测试成功")
                    successful_tests += 1
                else:
                    logger.warning(f"模型 {model_name} 测试失败: 所有重试都失败")
                    print(f"❌ 模型 {model_name} 测试失败: 所有重试都失败，返回 None")
            else:
                # 遍历该提供商的所有模型
                for model_name in models:
                    total_tests += 1
                    print(f"\n🤖 测试模型: {model_name}")
                    print("." * 40)
                    
                    # 调用聊天接口
                    response = manager.chat(
                        model=model_name,
                        message=test_message,
                        system_message=system_message,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    if response is not None:
                        logger.info(f"模型 {model_name} 测试成功，响应长度: {len(response)} 字符")
                        print(f"📤 请求成功")
                        print(f"📥 响应内容:")
                        print(f"   {response}")
                        print(f"✅ 模型 {model_name} 测试成功")
                        successful_tests += 1
                    else:
                        logger.warning(f"模型 {model_name} 测试失败: 所有重试都失败")
                        print(f"❌ 模型 {model_name} 测试失败: 所有重试都失败，返回 None")
            
            # 关闭管理器
            manager.close()
            
        except Exception as e:
            logger.error(f"提供商 {provider_name} 初始化失败: {str(e)}")
            print(f"❌ 提供商 {provider_name} 初始化失败: {str(e)}")
            # 如果提供商初始化失败，跳过该提供商的所有模型
            for _ in models:
                total_tests += 1
    
    # 显示测试总结
    success_rate = (successful_tests/total_tests*100) if total_tests > 0 else 0
    logger.info(f"测试完成 - 总数: {total_tests}, 成功: {successful_tests}, 失败: {total_tests - successful_tests}, 成功率: {success_rate:.1f}%")
    
    print("\n" + "=" * 80)
    print("🏁 测试完成总结")
    print("=" * 80)
    print(f"📊 总测试数: {total_tests}")
    print(f"✅ 成功测试: {successful_tests}")
    print(f"❌ 失败测试: {total_tests - successful_tests}")
    print(f"📈 成功率: {success_rate:.1f}%")
    
    if successful_tests == 0:
        print("\n💡 提示:")
        print("1. 请确保在 .env 文件中配置了相应的 API 密钥")
        print("2. 检查网络连接是否正常")
        print("3. 确认 API 密钥有足够的余额和权限")
        print("4. 检查配置文件中的模型名称是否正确")
    elif successful_tests < total_tests:
        print(f"\n⚠️  有 {total_tests - successful_tests} 个测试失败，请检查相关配置")
    else:
        print("\n🎉 所有测试都成功完成！")

if __name__ == "__main__":
    # 创建日志记录器
    logger = Logger("logs", "swe_bench_processor")
    logger.info("开始 SWE-bench 数据处理和 LLM API 测试程序")
    
    try:
        # 加载配置文件
        config = load_config(os.getenv("CONFIG_PATH", "config/config.yaml"))
        logger.info(f"成功加载配置文件")
        
        # 检查运行模式
        run_mode = os.getenv("RUN_MODE", "embedding").lower()  # 默认运行数据处理模式
        
        if run_mode == "data":
            logger.info("运行模式: SWE-bench 数据处理")
            # 处理 SWE-bench 数据集
            swe_loader, result = load_swe_bench_data(config, logger)
        elif run_mode == "llm":
            logger.info("运行模式: LLM API 测试")
            # 运行 LLM 测试
            test_provider_models(config, logger)
        elif run_mode == "embedding":
            logger.info("运行模式: Embedding 功能测试")
            # 运行 Embedding 测试
            test_embedding_functionality(config, logger)
        elif run_mode == "image_builder":
            logger.info("运行模式: SWE-bench Image Builder 测试")
            # 运行 Image Builder 测试
            # 从环境变量获取测试参数
            test_instance_ids = os.getenv("TEST_INSTANCE_IDS", "django__django-10914,astropy__astropy-12907").split(",")
            dataset_name = os.getenv("TEST_DATASET_NAME", "SWE-bench/SWE-bench_Lite")
            split = os.getenv("TEST_SPLIT", "test")
            max_workers = calculate_optimal_max_workers()
            
            force_rebuild = os.getenv("TEST_FORCE_REBUILD", "false").lower() == "true"
            timeout = int(os.getenv("TEST_TIMEOUT", "600"))
            
            results = test_swe_image_builder(
                instance_ids=test_instance_ids,
                dataset_name=dataset_name,
                split=split,
                max_workers=max_workers,
                force_rebuild=force_rebuild,
                timeout=timeout,
            )
            
            # 显示结果摘要
            total_instances = len(test_instance_ids)
            successful_instances = len([r for r in results["instance_results"].values() if r.get("success", False)])
            print(f"\n📊 测试完成: {successful_instances}/{total_instances} 成功")
        elif run_mode == "both":
            logger.info("运行模式: 数据处理 + LLM 测试")
            # 先处理数据
            swe_loader, result = load_swe_bench_data(config, logger)
            # 再运行 LLM 测试
            test_provider_models(config, logger)
        elif run_mode == "all":
            logger.info("运行模式: 全部功能测试")
            # 运行所有测试
            swe_loader, result = load_swe_bench_data(config, logger)
            test_provider_models(config, logger)
            test_embedding_functionality(config, logger)
            test_instance_ids = os.getenv("TEST_INSTANCE_IDS", "django__django-10914").split(",")
            max_workers = calculate_optimal_max_workers()
            results = test_swe_image_builder(
                instance_ids=test_instance_ids,
                dataset_name=os.getenv("TEST_DATASET_NAME", "SWE-bench/SWE-bench_Lite"),
                split=os.getenv("TEST_SPLIT", "test"),
                max_workers=max_workers,
                force_rebuild=os.getenv("TEST_FORCE_REBUILD", "false").lower() == "true",
                timeout=int(os.getenv("TEST_TIMEOUT", "600")),
            )
        else:
            logger.warning(f"未知运行模式: {run_mode}，默认运行数据处理")
            swe_loader, result = load_swe_bench_data(config, logger)
        
    except FileNotFoundError as e:
        logger.error(f"配置文件错误: {e}")
        print(f"❌ 配置文件错误: {e}")
        print("请确保在 src/config/ 目录下有 config.yaml 文件")
    except yaml.YAMLError as e:
        logger.error(f"YAML 解析错误: {e}")
        print(f"❌ YAML 解析错误: {e}")
        print("请检查 config.yaml 文件格式是否正确")
    except Exception as e:
        logger.error(f"程序执行错误: {e}")
        print(f"❌ 程序执行错误: {e}")
    finally:
        logger.info("SWE-bench 处理程序结束")
        logger.close()