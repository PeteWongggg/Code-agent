import asyncio
import sys
import os
from pathlib import Path
from typing import Optional
import logging

# Add the necessary paths to sys.path
sys.path.append('/Users/hengzhi/Code-agent')
sys.path.append('/Users/hengzhi/Code-agent/src')

# Import required modules
from src.tools.edit_tool import TextEditorTool
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


class EditToolTest:    
    def __init__(self, workspace_dir: str = "/tmp/edit_tool_test"):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.executor: Optional[Executor] = None
        self.edit_tool: Optional[TextEditorTool] = None
        self.session_id: Optional[str] = None
        
        self.image_builder: Optional[SWEBenchImageBuilder] = None
        self.current_instance: Optional[SWEbenchInstance] = None
        self.image_name: Optional[str] = None
        
        self.test_results = {}
        
    async def setup_swebench_image(self) -> bool:
        """Set up SWE-bench Docker image for testing"""
        try:
            dataset = load_swebench_dataset("princeton-nlp/SWE-bench_Lite", "dev")
            self.current_instance = dataset[1]
            self.image_builder = SWEBenchImageBuilder(
                dataset_name="princeton-nlp/SWE-bench_Lite",
                split="dev",
                instance_ids=[self.current_instance['instance_id']],
                max_workers=1,
                force_rebuild=False,  # Don't rebuild if image already exists
                namespace=None,
                tag="latest",
                env_image_tag="latest"  # Add required env_image_tag parameter
            )
            self.image_name = self.image_builder.get_image_name(self.current_instance['instance_id'])
            logger.info(f"✓ SWE-bench image setup completed: {self.image_name}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to setup SWE-bench image: {e}")
            return False
    
    def setup_executor_and_edit_tool(self) -> bool:
        """Set up executor and edit tool"""
        try:
            if not self.image_name:
                logger.error("No Docker image available. Run setup_swebench_image() first.")
                return False
            
            self.executor = Executor(self.image_name)
            # Initialize a session for the executor
            self.session_id = self.executor.init_session()
            if not self.session_id:
                logger.error("Failed to initialize executor session")
                return False
            
            self.edit_tool = TextEditorTool(model_provider="openai", executor=self.executor)
            logger.info("✓ Executor and EditTool setup completed")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to setup executor and edit tool: {e}")
            return False
    
    async def test_container_file_operations(self) -> bool:
        """Test basic container file read/write operations"""
        try:
            logger.info("Testing container file operations...")
            
            # Test file path in container
            test_file_path = Path("/testbed/test_file.txt")
            test_content = "\n\nHello, this is a test file!\nLine 2\nLine 3"
            
            # Test writing to container
            self.edit_tool.container_write_file(test_file_path, test_content, self.session_id)
            logger.info("✓ Container file write operation completed")
            
            # Test reading from container
            read_content = self.edit_tool.container_read_file(test_file_path, self.session_id)
            if read_content == test_content:
                logger.info("✓ Container file read operation successful")
                self.test_results["container_file_ops"] = True
                return True
            else:
                logger.error(f"✗ Container file read failed. Expected: {test_content}, Got: {read_content}")
                self.test_results["container_file_ops"] = False
                return False
                
        except Exception as e:
            logger.error(f"✗ Container file operations test failed: {e}")
            self.test_results["container_file_ops"] = False
            return False
    
    async def test_container_string_replacement(self) -> bool:
        """Test container string replacement operations"""
        try:
            logger.info("Testing container string replacement...")
            
            # Create a test file with content
            test_file_path = Path("/testbed/replace_test.txt")
            original_content = """def hello_world():
    print("Hello, World!")
    return "success"

def test_function():
    print("This is a test")
    return True"""
            
            # Write initial content
            self.edit_tool.container_write_file(test_file_path, original_content, self.session_id)
            
            # Test string replacement
            old_str = '    print("Hello, World!")'
            new_str = '    print("Hello, Container!")'
            replaced_content = original_content.replace(old_str, new_str)
            result = self.edit_tool.container_str_replace(test_file_path, old_str, new_str, self.session_id)
            
            # Read the file to verify the change
            updated_content = self.edit_tool.container_read_file(test_file_path, self.session_id)
            
            if updated_content == replaced_content:
                logger.info("✓ Container string replacement successful")
                self.test_results["container_str_replace"] = True
                return True
            else:
                logger.error(f"✗ Container string replacement failed. Content: {updated_content}")
                self.test_results["container_str_replace"] = False
                return False
                
        except Exception as e:
            logger.error(f"✗ Container string replacement test failed: {e}")
            self.test_results["container_str_replace"] = False
            return False
    
    async def test_container_multiple_replacements(self) -> bool:
        """Test multiple string replacements in container"""
        try:
            logger.info("Testing multiple container string replacements...")
            
            test_file_path = Path("/testbed/multi_replace_test.txt")
            original_content = """class TestClass:
    def method1(self):
        return "old_value1"
    
    def method2(self):
        return "old_value2"
    
    def method3(self):
        return "old_value3" """
            
            # Write initial content
            self.edit_tool.container_write_file(test_file_path, original_content, self.session_id)
            
            # Perform multiple replacements
            replacements = [
                ('return "old_value1"', 'return "new_value1"'),
                ('return "old_value2"', 'return "new_value2"'),
                ('return "old_value3"', 'return "new_value3"')
            ]
            
            for old_str, new_str in replacements:
                self.edit_tool.container_str_replace(test_file_path, old_str, new_str, self.session_id)
            
            # Verify all changes
            final_content = self.edit_tool.container_read_file(test_file_path, self.session_id)
            
            success = all(f'new_value{i}' in final_content for i in range(1, 4))
            success = success and not any(f'old_value{i}' in final_content for i in range(1, 4))
            
            if success:
                logger.info("✓ Multiple container string replacements successful")
                self.test_results["container_multi_replace"] = True
                return True
            else:
                logger.error(f"✗ Multiple container string replacements failed. Content: {final_content}")
                self.test_results["container_multi_replace"] = False
                return False
                
        except Exception as e:
            logger.error(f"✗ Multiple container string replacements test failed: {e}")
            self.test_results["container_multi_replace"] = False
            return False
    
    async def test_container_error_handling(self) -> bool:
        """Test error handling for container operations"""
        try:
            logger.info("Testing container error handling...")
            
            # Test reading non-existent file
            non_existent_path = Path("/testbed/non_existent_file.txt")
            try:
                self.edit_tool.container_read_file(non_existent_path, self.session_id)
                logger.error("✗ Should have failed reading non-existent file")
                self.test_results["container_error_handling"] = False
                return False
            except Exception:
                logger.info("✓ Correctly handled non-existent file read")
            
            # Test replacing non-existent string
            test_file_path = Path("/testbed/error_test.txt")
            self.edit_tool.container_write_file(test_file_path, "simple content", self.session_id)
            
            try:
                self.edit_tool.container_str_replace(test_file_path, "non_existent_string", "new_string", self.session_id)
                logger.error("✗ Should have failed replacing non-existent string")
                self.test_results["container_error_handling"] = False
                return False
            except Exception:
                logger.info("✓ Correctly handled non-existent string replacement")
            
            self.test_results["container_error_handling"] = True
            return True
            
        except Exception as e:
            logger.error(f"✗ Container error handling test failed: {e}")
            self.test_results["container_error_handling"] = False
            return False
    
    async def test_container_large_file_operations(self) -> bool:
        """Test operations on larger files in container"""
        try:
            logger.info("Testing container large file operations...")
            
            # Create a larger test file
            test_file_path = Path("/testbed/large_file_test.txt")
            lines = []
            for i in range(10000):
                lines.append(f"Line {i}: This is test content for line {i}")
            large_content = "\n".join(lines)
            
            # Write large file
            self.edit_tool.container_write_file(test_file_path, large_content, self.session_id)
            
            # Read it back
            logger.info("About to read large file from container...")
            read_content = self.edit_tool.container_read_file(test_file_path, self.session_id)
            logger.info(f"Finished reading large file. Got {len(read_content)} characters")
            
            # Get actual line count from file using wc -l
            try:
                return_code, wc_output = self.executor.execute(self.session_id, f"wc -l {test_file_path}")
                logger.info(f"wc -l output: '{wc_output}'")
                if return_code == 0:
                    # Parse the output - wc -l returns "number filename"
                    parts = wc_output.strip().split()
                    if parts:
                        actual_line_count = int(parts[0])
                        logger.info(f"Actual line count from wc -l: {actual_line_count}")
                    else:
                        logger.warning(f"Empty output from wc -l: {wc_output}")
                        actual_line_count = None
                else:
                    logger.warning(f"Failed to get line count with wc -l: {wc_output}")
                    actual_line_count = None
            except Exception as e:
                logger.warning(f"Failed to get line count with wc -l: {e}")
                actual_line_count = None
            
            read_line_count = len(read_content.split('\n'))
            logger.info(f"Read content line count: {read_line_count}")
            
            if read_content == large_content:
                logger.info("✓ Container large file operations successful")
                self.test_results["container_large_file"] = True
                return True
            else:
                logger.error(f"✗ Container large file operations failed - content mismatch")
                logger.error(f"Expected {len(large_content.split('\n'))} lines, got {read_line_count} lines from read_content")
                if actual_line_count is not None:
                    logger.error(f"Actual file has {actual_line_count} lines according to wc -l")
                
                # Debug: Show first and last few characters to see the difference
                logger.error(f"Expected content length: {len(large_content)}")
                logger.error(f"Read content length: {len(read_content)}")
                logger.error(f"Expected first 100 chars: {repr(large_content[:100])}")
                logger.error(f"Read first 100 chars: {repr(read_content[:100])}")
                logger.error(f"Expected last 100 chars: {repr(large_content[-100:])}")
                logger.error(f"Read last 100 chars: {repr(read_content[-100:])}")
                
                # Find the exact difference by comparing character by character
                min_len = min(len(large_content), len(read_content))
                diff_positions = []
                
                # Check from the beginning
                for i in range(min_len):
                    if large_content[i] != read_content[i]:
                        diff_positions.append(i)
                        if len(diff_positions) >= 5:  # Limit to first 5 differences
                            break
                
                if diff_positions:
                    logger.error(f"First differences found at positions: {diff_positions}")
                    for pos in diff_positions[:3]:  # Show first 3 differences
                        start = max(0, pos - 10)
                        end = min(len(large_content), pos + 10)
                        logger.error(f"Position {pos}:")
                        logger.error(f"  Expected: {repr(large_content[start:end])}")
                        logger.error(f"  Read:     {repr(read_content[start:end])}")
                
                # Check if read_content is shorter
                if len(read_content) < len(large_content):
                    logger.error(f"Read content is {len(large_content) - len(read_content)} characters shorter")
                    logger.error(f"Missing from end: {repr(large_content[len(read_content):])}")
                elif len(read_content) > len(large_content):
                    logger.error(f"Read content is {len(read_content) - len(large_content)} characters longer")
                    logger.error(f"Extra at end: {repr(read_content[len(large_content):])}")
                
                self.test_results["container_large_file"] = False
                return False
                
        except Exception as e:
            logger.error(f"✗ Container large file operations test failed: {e}")
            self.test_results["container_large_file"] = False
            return False
    
    def print_test_results(self):
        """Print comprehensive test results"""
        print(f"\n" + "="*80)
        print("EDIT TOOL CONTAINER OPERATIONS TEST RESULTS")
        print("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        print(f"\nOverall Results: {passed_tests}/{total_tests} tests passed")
        print("-" * 50)
        
        for test_name, result in self.test_results.items():
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print("-" * 50)
        if passed_tests == total_tests:
            print("🎉 All tests passed! EditTool works correctly with Executor.")
        else:
            print(f"⚠️  {total_tests - passed_tests} test(s) failed. Check the logs above for details.")
    
    async def cleanup(self):
        """Clean up resources"""
        if self.session_id and self.executor:
            self.executor.close_session(self.session_id)
        if self.executor:
            self.executor.shutdown()
    
    async def run_all_tests(self) -> bool:
        """Run all edit tool tests"""
        try:
            logger.info("Starting EditTool container operations tests...")
            
            # Setup
            if not await self.setup_swebench_image():
                return False
            
            if not self.setup_executor_and_edit_tool():
                return False
            
            # Run tests
            all_passed = True
            
            try:
                result = await self.test_container_file_operations()
                if not result:
                    all_passed = False
            except Exception as e:
                logger.error(f"✗ Container file operations test failed with exception: {e}")
                all_passed = False
            
            try:
                result = await self.test_container_string_replacement()
                if not result:
                    all_passed = False
            except Exception as e:
                logger.error(f"✗ Container string replacement test failed with exception: {e}")
                all_passed = False
            
            try:
                result = await self.test_container_multiple_replacements()
                if not result:
                    all_passed = False
            except Exception as e:
                logger.error(f"✗ Container multiple replacements test failed with exception: {e}")
                all_passed = False
            
            try:
                result = await self.test_container_error_handling()
                if not result:
                    all_passed = False
            except Exception as e:
                logger.error(f"✗ Container error handling test failed with exception: {e}")
                all_passed = False
            
            try:
                result = await self.test_container_large_file_operations()
                if not result:
                    all_passed = False
            except Exception as e:
                logger.error(f"✗ Container large file operations test failed with exception: {e}")
                all_passed = False
            
            self.print_test_results()
            return all_passed
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return False
        finally:
            await self.cleanup()


async def main():
    """Main test runner"""
    test = EditToolTest()    
    success = await test.run_all_tests()
    
    if success:
        print("\n🎉 All EditTool container operations tests completed successfully!")
    else:
        print("\n⚠️  Some EditTool container operations tests failed. Check the output above.")


if __name__ == "__main__":
    asyncio.run(main())
