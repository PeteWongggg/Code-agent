#!/usr/bin/env python3
"""
SWE-bench Workflow Script

This script demonstrates the complete SWE-bench evaluation workflow:
1. Get an instance from SWE-bench dataset
2. Build Docker image for the instance
3. Extract FAIL_TO_PASS test script
4. Run test with test patch applied (should fail)
5. Read the corresponding solution patch (gold patch)
6. Apply both test patch and solution patch
7. Run the test script again (should pass)
8. Compare outputs to verify correct evaluation

Key concepts:
- Test patch: Adds the failing test case that demonstrates the bug
- Solution patch: Fixes the bug in the source code
- FAIL_TO_PASS: Test should fail before solution, pass after solution
"""

import os
import sys
import json
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Add the necessary paths to sys.path
sys.path.append('/Users/hengzhizhang/SWE-bench')
sys.path.append('/Users/hengzhizhang/Code-agent')
sys.path.append('/Users/hengzhizhang/trae-agent')

# Import required modules
from datasets import load_dataset
from swebench.harness.utils import load_swebench_dataset
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.constants import SWEbenchInstance

# Import our custom modules
from src.managers.image_builder.build_image import SWEBenchImageBuilder
from src.tools.docker_tool_executor import DockerToolExecutor
from trae_agent.agent.docker_manager import DockerManager
from trae_agent.tools.base import ToolCall, ToolResult


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
        
        # Initialize Docker tool executor (will be set up later)
        self.docker_executor = None
        
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
        print(f"Problem: {self.current_instance['problem_statement'][:200]}...")
        
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
            tag=tag
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
        
        # Initialize Docker tool executor
        self.docker_executor = DockerToolExecutor(
            original_executor=None,  # We'll handle tool calls directly
            docker_manager=self.docker_manager,
            docker_tools=["bash", "str_replace_based_edit_tool", "json_edit_tool"],
            host_workspace_dir=str(self.workspace_dir),
            container_workspace_dir="/testbed"
        )
        
        print("Docker executor setup complete")
    
    def extract_test_script(self) -> List[str]:
        """Extract FAIL_TO_PASS test script from the instance"""
        if not self.current_instance:
            raise ValueError("No instance selected. Call get_instance() first.")
            
        # Create test spec to get the test scripts
        self.test_spec = make_test_spec(self.current_instance)
        
        # Get FAIL_TO_PASS tests
        fail_to_pass_tests = self.test_spec.FAIL_TO_PASS
        
        print(f"Extracted {len(fail_to_pass_tests)} FAIL_TO_PASS tests")
        for i, test in enumerate(fail_to_pass_tests):
            print(f"  Test {i+1}: {test}")
            
        return fail_to_pass_tests
    
    def run_test_script(self, test_commands: List[str], patch_applied: bool = False) -> TestResults:
        """Run the test script using docker_tool_executor following SWE-bench evaluation logic"""
        if not self.current_instance:
            raise ValueError("No instance selected. Call get_instance() first.")
        if not self.docker_executor:
            raise ValueError("Docker executor not set up. Call setup_docker_executor() first.")
            
        print(f"Running test script ({'after' if patch_applied else 'before'} patch application)...")
        
        # Get the test spec to understand the proper evaluation sequence
        if not self.test_spec:
            self.test_spec = make_test_spec(self.current_instance)
        
        # For FAIL_TO_PASS tests, we need to follow SWE-bench evaluation logic:
        # 1. Apply test patch (adds the failing test)
        # 2. Run the test (should fail)
        # 3. Apply solution patch (fixes the issue)
        # 4. Run the test again (should pass)
        
        if not patch_applied:
            # Before patch: Apply test patch and run test (should fail)
            test_script = self._create_before_patch_script()
        else:
            # After patch: Run test with both test patch and solution patch applied (should pass)
            test_script = self._create_after_patch_script()
        
        # Create tool call for bash execution
        tool_call = ToolCall(
            call_id="test_execution",
            name="bash",
            arguments={"command": test_script}
        )
        
        # Execute the test
        result = self.docker_executor._execute_in_docker(tool_call)
        
        print(f"Test execution {'succeeded' if result.success else 'failed'}")
        if not result.success:
            print(f"Error: {result.error}")
        
        return TestResults(
            before_patch=result.result if not patch_applied else None,
            after_patch=result.result if patch_applied else None,
            success_before=result.success if not patch_applied else False,
            success_after=result.success if patch_applied else False
        )
    
    def _create_before_patch_script(self) -> str:
        """Create the test script for before patch (should fail)"""
        # This follows the SWE-bench evaluation logic for FAIL_TO_PASS tests
        test_patch = self.current_instance["test_patch"]
        base_commit = self.current_instance["base_commit"]
        
        # Get test files from the test patch
        test_files = self._get_modified_files(test_patch)
        
        # Create the script that applies test patch and runs tests
        script_lines = [
            "cd /testbed",
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
            f"git config --global --add safe.directory /testbed",
            # Reset test files to base commit state
            f"git checkout {base_commit} {' '.join(test_files)}" if test_files else "echo 'No test files to reset'",
            # Apply the test patch (this adds the failing test)
            f"git apply -v - <<'EOF_TEST_PATCH'\n{test_patch}\nEOF_TEST_PATCH",
            # Run the test (should fail because the bug hasn't been fixed yet)
            f": '{self._START_TEST_OUTPUT}'",
            f"python -m pytest {' '.join(self.current_instance['FAIL_TO_PASS'])} -v",
            f": '{self._END_TEST_OUTPUT}'",
            # Revert test files back to base commit state
            f"git checkout {base_commit} {' '.join(test_files)}" if test_files else "echo 'No test files to reset'"
        ]
        
        return "\n".join(script_lines)
    
    def _create_after_patch_script(self) -> str:
        """Create the test script for after patch (should pass)"""
        # This runs the test with both test patch and solution patch already applied
        # The solution patch should have been applied in the apply_patch method
        script_lines = [
            "cd /testbed",
            "source /opt/miniconda3/bin/activate", 
            "conda activate testbed",
            f": '{self._START_TEST_OUTPUT}'",
            f"python -m pytest {' '.join(self.current_instance['FAIL_TO_PASS'])} -v",
            f": '{self._END_TEST_OUTPUT}'"
        ]
        
        return "\n".join(script_lines)
    
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
    
    # Constants for test output markers (from SWE-bench)
    _START_TEST_OUTPUT = "START_TEST_OUTPUT"
    _END_TEST_OUTPUT = "END_TEST_OUTPUT"
    
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
        if not self.docker_executor:
            raise ValueError("Docker executor not set up. Call setup_docker_executor() first.")
            
        print("Applying solution patch...")
        
        # Get the test spec to understand the proper evaluation sequence
        if not self.test_spec:
            self.test_spec = make_test_spec(self.current_instance)
        
        # For SWE-bench evaluation, we need to:
        # 1. Apply the test patch first (adds the failing test)
        # 2. Then apply the solution patch (fixes the issue)
        
        test_patch = self.current_instance["test_patch"]
        base_commit = self.current_instance["base_commit"]
        test_files = self._get_modified_files(test_patch)
        
        # Create the complete patch application script
        apply_script = "\n".join([
            "cd /testbed",
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
            f"git config --global --add safe.directory /testbed",
            # Reset test files to base commit state
            f"git checkout {base_commit} {' '.join(test_files)}" if test_files else "echo 'No test files to reset'",
            # Apply the test patch (adds the failing test)
            f"git apply -v - <<'EOF_TEST_PATCH'\n{test_patch}\nEOF_TEST_PATCH",
            # Apply the solution patch (fixes the issue)
            f"git apply -v - <<'EOF_SOLUTION_PATCH'\n{patch_content}\nEOF_SOLUTION_PATCH"
        ])
        
        tool_call = ToolCall(
            call_id="patch_application",
            name="bash",
            arguments={"command": apply_script}
        )
        
        # Execute patch application
        result = self.docker_executor._execute_in_docker(tool_call)
        
        print(f"Solution patch application {'succeeded' if result.success else 'failed'}")
        if not result.success:
            print(f"Error: {result.error}")
            print(f"Output: {result.result}")
        
        return result.success

def main():
    """Main function to run the workflow"""
    # Create workflow instance
    workflow = SWEBenchWorkflow()
    
    try:
        # Run complete workflow
        workflow.get_instance(instance_id=instance_id)
        # Step 2: Build image
        image_name = workflow.build_image()
        # Step 3: Set up Docker executor
        workflow.setup_docker_executor()
        # Step 4: Extract test script
        test_commands = workflow.extract_test_script()
        # Step 5: Run initial test (with test patch applied, should fail)
        results = workflow.run_test_script(test_commands, patch_applied=False)
        # Step 6: Get solution patch
        solution_patch = workflow.get_gold_patch()
        # Step 7: Apply solution patch
        patch_success = workflow.apply_patch(solution_patch)
        if not patch_success:
            print("Warning: Solution patch application failed, but continuing with test...")
        # Step 8: Run test again (with both test patch and solution patch, should pass)
        after_results = workflow.run_test_script(test_commands, patch_applied=True)
        # Combine results
        results.after_patch = after_results.after_patch
        results.success_after = after_results.success_after
        
        # Print comparison
        print(f"\nBEFORE SOLUTION PATCH (with test patch only):")
        print(f"  Test Result: {'PASSED' if results.success_before else 'FAILED'}")
        print(f"  Output:")
        if results.before_patch:
            print("  " + "\n  ".join(results.before_patch.split('\n')[:20]))
            if len(results.before_patch.split('\n')) > 20:
                print("  ... (truncated)")
        else:
            print("  No output")
        print(f"\nAFTER SOLUTION PATCH (with test patch + solution patch):")
        print(f"  Test Result: {'PASSED' if results.success_after else 'FAILED'}")
        print(f"  Output:")
        if results.after_patch:
            print("  " + "\n  ".join(results.after_patch.split('\n')[:20]))
            if len(results.after_patch.split('\n')) > 20:
                print("  ... (truncated)")
        else:
            print("  No output")
        
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
