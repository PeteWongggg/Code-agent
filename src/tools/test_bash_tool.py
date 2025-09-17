import asyncio
import sys
import os
from pathlib import Path
from typing import Optional, List
import logging

# Add the necessary paths to sys.path
sys.path.append('/Users/hengzhi/Code-agent')
sys.path.append('/Users/hengzhi/Code-agent/src')

# Import required modules
from src.tools.bash_tool import BashTool
from src.tools.executor import Executor
from src.managers.image_builder.build_image import SWEBenchImageBuilder

from datasets import load_dataset
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.utils import load_swebench_dataset
from swebench.harness.constants import SWEbenchInstance, MAP_REPO_VERSION_TO_SPECS, START_TEST_OUTPUT, END_TEST_OUTPUT
from swebench.harness.test_spec.python import get_test_directives
from swebench.harness import constants
SWEBENCH_AVAILABLE = True

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MinimalToolsTest:    
    def __init__(self, workspace_dir: str = "/tmp/tools_test"):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.executor: Optional[Executor] = None
        self.bash_tool: Optional[BashTool] = None
        
        self.image_builder: Optional[SWEBenchImageBuilder] = None
        self.current_instance: Optional[SWEbenchInstance] = None
        self.image_name: Optional[str] = None
        
        self.before_patch_output: Optional[str] = None
        self.after_patch_output: Optional[str] = None
        self.before_patch_success: bool = False
        self.after_patch_success: bool = False
        
    async def setup_swebench_image(self) -> bool:
        dataset = load_swebench_dataset("princeton-nlp/SWE-bench_Lite", "dev")
        self.current_instance = dataset[1]
        self.image_builder = SWEBenchImageBuilder(
            dataset_name="princeton-nlp/SWE-bench_Lite",
            split="dev",
            instance_ids=[self.current_instance['instance_id']],
            max_workers=1,
            force_rebuild=False,  # Don't rebuild if image already exists
            namespace= None,
            tag="latest",
            env_image_tag="latest"  # Add required env_image_tag parameter
        )
        self.image_name = self.image_builder.get_image_name(self.current_instance['instance_id'])
        return True
    
    def setup_executor_and_bash_tool(self) -> bool:
        if not self.image_name:
            logger.error("No Docker image available. Run setup_swebench_image() first.")
            return False
        self.executor = Executor(self.image_name)
        # Session '0' is automatically initialized by Executor.__init__
        self.bash_tool = BashTool(model_provider="openai", executor=self.executor)            
        return True
    
    def _get_modified_files(self, patch_content: str) -> List[str]:
        import re
        files = []
        for line in patch_content.split('\n'):
            if line.startswith('diff --git a/'):
                # Extract the file path after 'a/'
                match = re.search(r'diff --git a/([^ ]+)', line)
                if match:
                    files.append(match.group(1))
        return files
    
    async def test_swebench_workflow_commands(self) -> bool:
        if not self.current_instance or not self.bash_tool:
            logger.error("SWE-bench instance or BashTool not available")
            return False
            
        try:
            test_spec = make_test_spec(self.current_instance, env_image_tag="latest")
            repo = self.current_instance["repo"]
            version = self.current_instance["version"]
            base_test_cmd = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
            test_directives = get_test_directives(self.current_instance)
            
            if isinstance(base_test_cmd, list):
                base_test_cmd = base_test_cmd[-1]
            
            if test_directives:
                test_command = f"{base_test_cmd} {' '.join(test_directives)}"
            else:
                test_command = base_test_cmd
            
            test_patch = self.current_instance["test_patch"]
            base_commit = self.current_instance["base_commit"]
            test_files = self._get_modified_files(test_patch)

            script_lines = [
                "cd /testbed",
                "source /opt/miniconda3/bin/activate",
                "conda activate testbed",
                f"git config --global --add safe.directory /testbed",
                # Apply the test patch (this adds the failing test)
                f"git apply -v - <<'EOF_TEST_PATCH'\n{test_patch}\nEOF_TEST_PATCH",
                f": '{START_TEST_OUTPUT}'",
                test_command,
                f": '{END_TEST_OUTPUT}'",
                f"git checkout {base_commit} {' '.join(test_files)}" if test_files else "echo 'No test files to reset'"
            ]
            test_script_before = "\n".join(script_lines)

            result_before = await self.bash_tool.container_execute({
                "command": test_script_before,
                "restart": False
            })
            
            self.before_patch_output = result_before.output
            self.before_patch_success = (result_before.error_code == 0)
            
            # Test 2: Apply solution patch and run test again (should pass)
            solution_patch = self.current_instance.get('patch', '')
            
            test_script_after = f"""
cd /testbed
source /opt/miniconda3/bin/activate
conda activate testbed
git config --global --add safe.directory /testbed
git apply -v - <<'EOF_TEST_PATCH'\n{test_patch}\nEOF_TEST_PATCH
git apply -v - <<'EOF_SOLUTION_PATCH'\n{solution_patch}\nEOF_SOLUTION_PATCH
: '{START_TEST_OUTPUT}'
{test_command}
: '{END_TEST_OUTPUT}'
"""
            
            result_after = await self.bash_tool.container_execute({"command": test_script_after, "restart": False})
            
            self.after_patch_output = result_after.output
            self.after_patch_success = (result_after.error_code == 0)
            
            return True
            
        except Exception as e:
            logger.error(f"SWE-bench workflow test failed: {e}")
            return False
    
    def print_test_results(self):
        print(f"\nBEFORE SOLUTION PATCH (with test patch only):" + "*" * 80)
        print(f"  Test Result: {'PASSED' if self.before_patch_success else 'FAILED'}")
        print(f"  Output:")
        if self.before_patch_output:
            print(self.before_patch_output)
        else:
            print("No output available")
        
        print(f"\nAFTER SOLUTION PATCH (with test patch + solution patch):" + "*" * 80)
        print(f"  Test Result: {'PASSED' if self.after_patch_success else 'FAILED'}")
        print(f"  Output:")
        if self.after_patch_output:
            print(self.after_patch_output)
        else:
            print("No output available")
    
    async def cleanup(self):
        if self.bash_tool:
            await self.bash_tool.close()            
        if self.executor:
            self.executor.shutdown()
    
    async def run_all_tests(self) -> bool:
        try:
            await self.setup_swebench_image()
            self.setup_executor_and_bash_tool()
            all_passed = True            
            try:
                result = await self.test_swebench_workflow_commands()
                if not result:
                    logger.error(f"✗ SWE-bench Workflow FAILED")
                    all_passed = False
            except Exception as e:
                logger.error(f"✗ SWE-bench Workflow FAILED with exception: {e}")
                all_passed = False
            self.print_test_results()
            return all_passed
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return False
        finally:
            await self.cleanup()

async def main():
    test = MinimalToolsTest()    
    success = await test.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
