from pathlib import Path
import time
import traceback
from typing import Dict, Any, List, Optional
import docker
import json

# Import the SWEBenchImageBuilder class
from build_image import SWEBenchImageBuilder

# Import from the installed swebench package
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
    
    Args:
        instance_ids: List of instance IDs to build and test
        dataset_name: Name of the dataset to use
        split: Split to use (dev/test)
        max_workers: Number of workers for parallel processing
        force_rebuild: Whether to force rebuild all images
        timeout: Timeout for test execution
        
    Returns:
        Dictionary with test results for all instances
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


def main():
    """Main function to test the SWEBenchImageBuilder."""
    # Test with 2 different instance IDs
    test_instance_ids = [
        "django__django-11680",  # The one from the original test
        "astropy__astropy-11693",  # Another instance to test
    ]    
    # Run the test
    results = test_swe_image_builder(
        instance_ids=test_instance_ids,
        dataset_name="SWE-bench/SWE-bench",
        split="test",
        max_workers=1,  # Build sequentially to avoid memory issues
        force_rebuild=False,  # Set to True if you want to rebuild existing images
        timeout=600,
    )


if __name__ == "__main__":
    main()
