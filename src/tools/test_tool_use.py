#!/usr/bin/env python3
"""
This script demonstrates the complete SWE-bench evaluation workflow:
1. Get an instance from SWE-bench dataset
2. Build Docker image for the instance
3. Extract FAIL_TO_PASS test script
4. Apply test patch and run test script (should fail)
5. Read the corresponding solution patch (gold patch)
6. Apply both solution patch
7. Run the test script again (should pass)
"""

import sys
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

# Add the necessary paths to sys.path
sys.path.append('/Users/hengzhizhang/SWE-bench')
sys.path.append('/Users/hengzhizhang/Code-agent')
sys.path.append('/Users/hengzhizhang/trae-agent')

# Import required modules
from datasets import load_dataset
from swebench.harness.utils import load_swebench_dataset
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.constants import SWEbenchInstance, MAP_REPO_VERSION_TO_SPECS, START_TEST_OUTPUT, END_TEST_OUTPUT
from swebench.harness.test_spec.python import get_test_directives

# Import our custom modulesc
from src.managers.image_builder.build_image import SWEBenchImageBuilder
from trae_agent.agent.docker_manager import DockerManager


@dataclass
class TestResults:
    """Container for test execution results"""
    before_patch: Optional[str] = None
    after_patch: Optional[str] = None
    success_before: bool = False
    success_after: bool = False


class SWEBenchWorkflow:
    """Main workflow class for SWE-bench evaluation"""
    
    def __init__(self, workspace_dir: str = "/tmp/swe_bench_workflow"):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Docker manager (will be set up later with the correct image)
        self.docker_manager = None
        
        self.current_instance: Optional[SWEbenchInstance] = None
        self.test_spec = None
        self.image_builder = None
        self.image_name = None
        
    def get_instance(self, dataset_name: str = "princeton-nlp/SWE-bench_Lite", 
                    split: str = "dev", instance_id: Optional[str] = None) -> SWEbenchInstance:
        """Get a specific instance from SWE-bench dataset"""
        print(f"Loading SWE-bench dataset: {dataset_name}, split: {split}")
        
        # Load the dataset
        dataset = load_swebench_dataset(dataset_name, split)
        
        if instance_id:
            # Find specific instance
            for instance in dataset:
                if instance['instance_id'] == instance_id:
                    self.current_instance = instance
                    break
            else:
                raise ValueError(f"Instance {instance_id} not found in dataset")
        else:
            # Get the second instance for demonstration (creates new test file)
            self.current_instance = dataset[1]  # sqlfluff-2419
            
        print(f"Selected instance: {self.current_instance['instance_id']}")
        print(f"Repository: {self.current_instance['repo']}")
        print(f"Problem: {self.current_instance['problem_statement']}...")
        
        return self.current_instance
    
    def build_image(self, namespace: str = "", tag: str = "latest") -> str:
        """Build Docker image for the current instance"""
        if not self.current_instance:
            raise ValueError("No instance selected. Call get_instance() first.")
            
        # Create image builder
        self.image_builder = SWEBenchImageBuilder(
            dataset_name="princeton-nlp/SWE-bench_Lite",
            split="dev",
            instance_ids=[self.current_instance['instance_id']],
            max_workers=1,
            force_rebuild=False,  # Don't rebuild if image already exists
            namespace=namespace if namespace else None,
            tag=tag,
            env_image_tag="latest"  # Add required env_image_tag parameter
        )
        
        # Get the image name
        image_name = self.image_builder.get_image_name(self.current_instance['instance_id'])
        
        # Check if image already exists
        import docker
        client = docker.from_env()
        try:
            client.images.get(image_name)
            print(f"Using existing Docker image: {image_name}")
        except docker.errors.ImageNotFound:
            print(f"Building Docker image for instance: {self.current_instance['instance_id']}")
            # The image builder will handle the actual building
            print(f"Built image: {image_name}")
        
        self.image_name = image_name
        return image_name
    
    def setup_docker_executor(self):
        """Set up Docker manager and executor with the built image"""
        if not self.image_name:
            raise ValueError("No image built. Call build_image() first.")
            
        print("Setting up Docker executor...")
        
        # Initialize Docker manager with the built image
        self.docker_manager = DockerManager(
            image=self.image_name,
            container_id=None,
            dockerfile_path=None,
            docker_image_file=None,
            workspace_dir=str(self.workspace_dir),
            tools_dir=None,
            interactive=True
        )
        
        # Start the Docker manager
        self.docker_manager.start()
        
        print("Docker executor setup complete")
    
    def extract_test_script(self) -> List[str]:
        """Extract test commands from the instance using proper SWE-bench methodology"""
        if not self.current_instance:
            raise ValueError("No instance selected. Call get_instance() first.")
            
        # Create test spec to get the test scripts
        self.test_spec = make_test_spec(self.current_instance, env_image_tag="latest")
        
        # Get the base test command for this repository
        repo = self.current_instance["repo"]
        version = self.current_instance["version"]
        base_test_cmd = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
        
        # Get test directives from the test patch (actual test files/functions)
        test_directives = get_test_directives(self.current_instance)
        
        # Construct the complete test command
        if isinstance(base_test_cmd, list):
            # Some repos have multiple test commands, use the last one
            base_test_cmd = base_test_cmd[-1]
        
        # Combine base command with test directives
        if test_directives:
            test_command = f"{base_test_cmd} {' '.join(test_directives)}"
        else:
            test_command = base_test_cmd
        
        print(f"Repository: {repo}")
        print(f"Base test command: {base_test_cmd}")
        print(f"Test directives: {test_directives}")
        print(f"Complete test command: {test_command}")
        
        return [test_command]
    
    def run_test_script(self, test_commands: List[str], patch_applied: bool = False) -> TestResults:
        """Run the test script using docker_tool_executor following SWE-bench evaluation logic"""
        if not self.current_instance:
            raise ValueError("No instance selected. Call get_instance() first.")
        
        # Get the test spec to understand the proper evaluation sequence
        if not self.test_spec:
            self.test_spec = make_test_spec(self.current_instance, env_image_tag="latest")
        
        # For FAIL_TO_PASS tests, we need to follow SWE-bench evaluation logic:
        # 1. Apply test patch (adds the failing test)
        # 2. Run the test (should fail)
        # 3. Apply solution patch (fixes the issue)
        # 4. Run the test again (should pass)
        
        if not patch_applied:
            # Before patch: Apply test patch and run test
            test_patch = self.current_instance["test_patch"]
            base_commit = self.current_instance["base_commit"]
            # Get test files from the test patch
            test_files = self._get_modified_files(test_patch)
            # Get the proper test command
            test_command = test_commands[0]
            # Create the script that applies test patch and runs tests
            script_lines = [
                "cd /testbed",
                "source /opt/miniconda3/bin/activate",
                "conda activate testbed",
                f"git config --global --add safe.directory /testbed",
                # Apply the test patch (this adds the failing test)
                f"git apply -v - <<'EOF_TEST_PATCH'\n{test_patch}\nEOF_TEST_PATCH",
                # Run the test (should fail because the bug hasn't been fixed yet)
                f": '{START_TEST_OUTPUT}'",
                test_command,
                f": '{END_TEST_OUTPUT}'",
                # Revert test files back to base commit state
                f"git checkout {base_commit} {' '.join(test_files)}" if test_files else "echo 'No test files to reset'"
            ]
            test_script = "\n".join(script_lines)
        else:
            test_command = test_commands[0]
            script_lines = [
                "cd /testbed",
                "source /opt/miniconda3/bin/activate", 
                "conda activate testbed",
                f": '{START_TEST_OUTPUT}'",
                test_command,
                f": '{END_TEST_OUTPUT}'"
            ]
            test_script = "\n".join(script_lines)
        
        # Execute the test
        exit_code, output = self.docker_manager.execute(test_script)

        return TestResults(
            before_patch=output if not patch_applied else None,
            after_patch=output if patch_applied else None,
            success_before=(exit_code == 0) if not patch_applied else False,
            success_after=(exit_code == 0) if patch_applied else False
        )
    
    def _get_modified_files(self, patch_content: str) -> List[str]:
        """Extract modified files from a patch"""
        import re
        files = []
        for line in patch_content.split('\n'):
            if line.startswith('diff --git a/'):
                # Extract the file path after 'a/'
                match = re.search(r'diff --git a/([^ ]+)', line)
                if match:
                    files.append(match.group(1))
        return files
    
    # Constants for test output markers are now imported from SWE-bench
    
    def get_gold_patch(self) -> str:
        """Read the corresponding gold patch from SWE-bench"""
        if not self.current_instance:
            raise ValueError("No instance selected. Call get_instance() first.")
            
        gold_patch = self.current_instance.get('patch', '')
        
        print(f"Retrieved gold patch ({len(gold_patch)} characters)")
        print("Patch preview:")
        print(gold_patch[:500] + "..." if len(gold_patch) > 500 else gold_patch)
        
        return gold_patch
    
    def apply_patch(self, patch_content: str) -> bool:
        """Apply the solution patch using docker_tool_executor following SWE-bench logic"""
        if not self.current_instance:
            raise ValueError("No instance selected. Call get_instance() first.")

        # Get the test spec to understand the proper evaluation sequence
        if not self.test_spec:
            self.test_spec = make_test_spec(self.current_instance, env_image_tag="latest")
        
        # For SWE-bench evaluation, we need to:
        # 1. Apply the test patch first (adds the failing test)
        # 2. Then apply the solution patch (fixes the issue)
        
        test_patch = self.current_instance["test_patch"]
        # Create the complete patch application script
        apply_script = "\n".join([
            "cd /testbed",
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
            f"git config --global --add safe.directory /testbed",
            # Apply the test patch (adds the failing test)
            f"git apply -v - <<'EOF_TEST_PATCH'\n{test_patch}\nEOF_TEST_PATCH",
            # Apply the solution patch (fixes the issue)
            f"git apply -v - <<'EOF_SOLUTION_PATCH'\n{patch_content}\nEOF_SOLUTION_PATCH"
        ])
        
        # Execute patch application
        exit_code, output = self.docker_manager.execute(apply_script)
        
        print(f"Solution patch application {'succeeded' if exit_code == 0 else 'failed'}")
        if exit_code != 0:
            print(f"Error: {output}")

        return exit_code == 0

def main():
    """Main function to run the workflow"""
    # Create workflow instance
    workflow = SWEBenchWorkflow()
    
    try:
        # Run complete workflow
        workflow.get_instance()  # Use default instance (sqlfluff-2419)
        # Step 2: Build image
        image_name = workflow.build_image()
        # Step 3: Set up Docker executor
        workflow.setup_docker_executor()
        # Step 4: Extract test script (now returns proper test commands)
        test_commands = workflow.extract_test_script()
        # Step 5: Run initial test (with test patch applied, should fail)
        results = workflow.run_test_script(test_commands, patch_applied=False)
        # Step 6: Get solution patch
        solution_patch = workflow.get_gold_patch()
        # Step 7: Apply solution patch
        patch_success = workflow.apply_patch(solution_patch)
        # Step 8: Run test again (with both test patch and solution patch, should pass)
        after_results = workflow.run_test_script(test_commands, patch_applied=True)
        # Combine results
        results.after_patch = after_results.after_patch
        results.success_after = after_results.success_after
        
        # Print comparison
        print(f"\nBEFORE SOLUTION PATCH (with test patch only):")
        print(f"  Test Result: {'PASSED' if results.success_before else 'FAILED'}")
        print(f"  Output:" + "*" * 100)
        print("  " + "\n  ".join(results.before_patch.split('\n')))
        print(f"\nAFTER SOLUTION PATCH (with test patch + solution patch):")
        print(f"  Test Result: {'PASSED' if results.success_after else 'FAILED'}")
        print(f"  Output:" + "*" * 100)
        print("  " + "\n  ".join(results.after_patch.split('\n')))
        
    except Exception as e:
        print(f"Error during workflow execution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\nCleaning up Docker resources...")
        if workflow.docker_manager:
            workflow.docker_manager.stop()


if __name__ == "__main__":
    main()
