import asyncio
import sys
import os
import time
import logging
from pathlib import Path
from typing import Optional, List
import docker
from docker.errors import DockerException, ImageNotFound

# Add the necessary paths to sys.path
sys.path.append('/Users/hengzhi/Code-agent')
sys.path.append('/Users/hengzhi/Code-agent/src')

# Import required modules
from src.tools.executor import Executor
from src.managers.image_builder.build_image import SWEBenchImageBuilder
from src.tools.bash_tool import BashTool

from datasets import load_dataset
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.utils import load_swebench_dataset
from swebench.harness.constants import SWEbenchInstance, MAP_REPO_VERSION_TO_SPECS, START_TEST_OUTPUT, END_TEST_OUTPUT
from swebench.harness.test_spec.python import get_test_directives
from swebench.harness import constants

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExecutorTestSuite:
    """
    Comprehensive test suite for the Executor class.
    Tests all functionalities including check_session, execute, session management, etc.
    """
    
    def __init__(self, workspace_dir: str = "/tmp/executor_test"):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.executor: Optional[Executor] = None
        self.bash_tool: Optional[BashTool] = None
        
        self.image_builder: Optional[SWEBenchImageBuilder] = None
        self.current_instance: Optional[SWEbenchInstance] = None
        self.image_name: Optional[str] = None
        
        self.test_results = {
            'image_building': False,
            'executor_initialization': False,
            'basic_command_execution': False,
            'session_management': False,
            'check_session_functionality': False,
            'multiple_sessions': False,
            'error_handling': False,
            'timeout_handling': False,
            'container_cleanup': False,
            'swebench_workflow': False
        }
        
    async def setup_swebench_image(self) -> bool:
        """Build a SWE-bench Docker image for testing."""
        try:
            logger.info("Setting up SWE-bench image...")
            dataset = load_swebench_dataset("princeton-nlp/SWE-bench_Lite", "dev")
            self.current_instance = dataset[1]  # Use the second instance for testing
            
            self.image_builder = SWEBenchImageBuilder(
                dataset_name="princeton-nlp/SWE-bench_Lite",
                split="dev",
                instance_ids=[self.current_instance['instance_id']],
                max_workers=1,
                force_rebuild=False,  # Don't rebuild if image already exists
                namespace=None,
                tag="latest",
                env_image_tag="latest"
            )
            
            self.image_name = self.image_builder.get_image_name(self.current_instance['instance_id'])
            logger.info(f"Successfully built/retrieved image: {self.image_name}")
            self.test_results['image_building'] = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup SWE-bench image: {e}")
            return False
    
    def setup_executor(self) -> bool:
        """Initialize the Executor with the built image."""
        try:
            if not self.image_name:
                logger.error("No Docker image available. Run setup_swebench_image() first.")
                return False
                
            logger.info(f"Initializing Executor with image: {self.image_name}")
            self.executor = Executor(self.image_name)
            self.bash_tool = BashTool(model_provider="openai", executor=self.executor)
            
            logger.info("Executor initialized successfully")
            self.test_results['executor_initialization'] = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Executor: {e}")
            return False
    
    def test_basic_command_execution(self) -> bool:
        """Test basic command execution functionality."""
        try:
            logger.info("Testing basic command execution...")
            
            # Test simple commands
            test_commands = [
                ("echo 'Hello World'", "Hello World"),
                ("pwd", "/workspace"),
                ("ls -la", None),  # Just check it doesn't fail
                ("whoami", "root"),
                ("date", None)  # Just check it doesn't fail
            ]
            
            for command, expected_output in test_commands:
                exit_code, output = self.executor.execute('0', command)
                
                if exit_code != 0:
                    logger.error(f"Command '{command}' failed with exit code {exit_code}")
                    return False
                
                if expected_output and expected_output not in output:
                    logger.error(f"Command '{command}' output doesn't contain expected text. Got: {output}")
                    return False
                
                logger.info(f"✓ Command '{command}' executed successfully")
            
            self.test_results['basic_command_execution'] = True
            return True
            
        except Exception as e:
            logger.error(f"Basic command execution test failed: {e}")
            return False
    
    def test_session_management(self) -> bool:
        """Test session management functionality."""
        try:
            logger.info("Testing session management...")
            
            # Test that default session '0' exists
            if '0' not in self.executor.sessions:
                logger.error("Default session '0' not found")
                return False
            
            # Test creating a new session
            new_session_id = self.executor.init_session()
            if not new_session_id:
                logger.error("Failed to create new session")
                return False
            
            # Test executing command in new session
            exit_code, output = self.executor.execute(new_session_id, "echo 'New session test'")
            if exit_code != 0 or "New session test" not in output:
                logger.error("Failed to execute command in new session")
                return False
            
            # Test closing the new session
            self.executor.close_session(new_session_id)
            if new_session_id in self.executor.sessions:
                logger.error("Session not properly closed")
                return False
            
            logger.info("✓ Session management test passed")
            self.test_results['session_management'] = True
            return True
            
        except Exception as e:
            logger.error(f"Session management test failed: {e}")
            return False
    
    def test_check_session_functionality(self) -> bool:
        """Test the check_session method specifically."""
        try:
            logger.info("Testing check_session functionality...")
            
            # Test with healthy session
            is_healthy = self.executor.check_session()
            if not is_healthy:
                logger.error("check_session returned False for healthy session")
                return False
            
            # Test that session is still working after check
            exit_code, output = self.executor.execute('0', "echo 'Session still working'")
            if exit_code != 0 or "Session still working" not in output:
                logger.error("Session not working after check_session")
                return False
            
            # Test with invalid session ID (should create new session)
            # First, let's manually break the session by removing it
            original_session = self.executor.sessions.get('0')
            if original_session:
                self.executor.sessions.pop('0')
            
            # Now test check_session - it should detect the missing session and create a new one
            is_healthy = self.executor.check_session()
            if not is_healthy:
                logger.error("check_session failed to recover from missing session")
                return False
            
            # Verify the new session works
            exit_code, output = self.executor.execute('0', "echo 'Recovered session test'")
            if exit_code != 0 or "Recovered session test" not in output:
                logger.error("Recovered session not working properly")
                return False
            
            logger.info("✓ check_session functionality test passed")
            self.test_results['check_session_functionality'] = True
            return True
            
        except Exception as e:
            logger.error(f"check_session functionality test failed: {e}")
            return False
    
    def test_multiple_sessions(self) -> bool:
        """Test handling multiple concurrent sessions."""
        try:
            logger.info("Testing multiple sessions...")
            
            # Create multiple sessions
            session_ids = []
            for i in range(3):
                session_id = self.executor.init_session()
                if not session_id:
                    logger.error(f"Failed to create session {i}")
                    return False
                session_ids.append(session_id)
            
            # Execute different commands in each session
            for i, session_id in enumerate(session_ids):
                command = f"echo 'Session {i} test'"
                exit_code, output = self.executor.execute(session_id, command)
                if exit_code != 0 or f"Session {i} test" not in output:
                    logger.error(f"Failed to execute command in session {session_id}")
                    return False
            
            # Test that sessions are independent
            # Set a variable in one session and check it doesn't affect others
            self.executor.execute(session_ids[0], "export TEST_VAR='session0'")
            self.executor.execute(session_ids[1], "export TEST_VAR='session1'")
            
            # Check that each session has its own variable
            exit_code, output = self.executor.execute(session_ids[0], "echo $TEST_VAR")
            if exit_code != 0 or "session0" not in output:
                logger.error("Session isolation failed - variable not set correctly")
                return False
            
            exit_code, output = self.executor.execute(session_ids[1], "echo $TEST_VAR")
            if exit_code != 0 or "session1" not in output:
                logger.error("Session isolation failed - variable not set correctly")
                return False
            
            # Clean up sessions
            for session_id in session_ids:
                self.executor.close_session(session_id)
            
            logger.info("✓ Multiple sessions test passed")
            self.test_results['multiple_sessions'] = True
            return True
            
        except Exception as e:
            logger.error(f"Multiple sessions test failed: {e}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling for various scenarios."""
        try:
            logger.info("Testing error handling...")
            
            # Test command that doesn't exist
            exit_code, output = self.executor.execute('0', "nonexistentcommand12345")
            if exit_code == 0:
                logger.error("Expected non-zero exit code for nonexistent command")
                return False
            
            # Test invalid session ID
            exit_code, output = self.executor.execute('nonexistent_session', "echo 'test'")
            if exit_code != -1:
                logger.error("Expected error for invalid session ID")
                return False
            
            # Test command that fails
            exit_code, output = self.executor.execute('0', "ls /nonexistent_directory_12345")
            if exit_code == 0:
                logger.error("Expected non-zero exit code for failed command")
                return False
            
            logger.info("✓ Error handling test passed")
            self.test_results['error_handling'] = True
            return True
            
        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
            return False
    
    def test_timeout_handling(self) -> bool:
        """Test timeout handling for long-running commands."""
        try:
            logger.info("Testing timeout handling...")
            
            # Test with a short timeout
            start_time = time.time()
            exit_code, output = self.executor.execute('0', "sleep 2", timeout=1)
            end_time = time.time()
            
            # Should timeout and return -1
            if exit_code != -1:
                logger.error("Expected timeout error for long-running command")
                return False
            
            # Should timeout within reasonable time (not wait the full 2 seconds)
            if end_time - start_time > 1.5:
                logger.error("Timeout took too long")
                return False
            
            logger.info("✓ Timeout handling test passed")
            self.test_results['timeout_handling'] = True
            return True
            
        except Exception as e:
            logger.error(f"Timeout handling test failed: {e}")
            return False
    
    async def test_swebench_workflow(self) -> bool:
        """Test SWE-bench workflow similar to test_bash_tool.py."""
        try:
            logger.info("Testing SWE-bench workflow...")
            
            if not self.current_instance or not self.bash_tool:
                logger.error("SWE-bench instance or BashTool not available")
                return False
            
            # Test basic SWE-bench commands
            test_commands = [
                "cd /testbed",
                "pwd",
                "ls -la",
                "git status",
                "python --version"
            ]
            
            for command in test_commands:
                result = await self.bash_tool.container_execute({
                    "command": command,
                    "restart": False
                })
                
                if result.error_code != 0:
                    logger.warning(f"Command '{command}' failed: {result.error}")
                else:
                    logger.info(f"✓ Command '{command}' executed successfully")
            
            # Test more complex SWE-bench workflow
            workflow_script = """
            cd /testbed
            source /opt/miniconda3/bin/activate
            conda activate testbed
            git config --global --add safe.directory /testbed
            echo "Workflow test completed"
            """
            
            result = await self.bash_tool.container_execute({
                "command": workflow_script,
                "restart": False
            })
            
            if result.error_code != 0:
                logger.error(f"SWE-bench workflow failed: {result.error}")
                return False
            
            logger.info("✓ SWE-bench workflow test passed")
            self.test_results['swebench_workflow'] = True
            return True
            
        except Exception as e:
            logger.error(f"SWE-bench workflow test failed: {e}")
            return False
    
    def test_container_cleanup(self) -> bool:
        """Test container cleanup functionality."""
        try:
            logger.info("Testing container cleanup...")
            
            # Test shutdown functionality
            if self.executor:
                self.executor.shutdown()
                
                # Verify container is stopped
                if self.executor.container:
                    try:
                        # Try to get container info - should fail if properly stopped
                        container_info = self.executor.container.attrs
                        if container_info.get('State', {}).get('Running', False):
                            logger.error("Container still running after shutdown")
                            return False
                    except Exception:
                        # This is expected - container should be removed
                        pass
            
            logger.info("✓ Container cleanup test passed")
            self.test_results['container_cleanup'] = True
            return True
            
        except Exception as e:
            logger.error(f"Container cleanup test failed: {e}")
            return False
    
    def print_test_results(self):
        """Print comprehensive test results."""
        print("\n" + "="*80)
        print("EXECUTOR TEST RESULTS")
        print("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        for test_name, result in self.test_results.items():
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print("-"*80)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print("="*80)
    
    async def run_all_tests(self) -> bool:
        """Run all tests in sequence."""
        try:
            logger.info("Starting Executor test suite...")
            
            # Setup phase
            if not await self.setup_swebench_image():
                logger.error("Failed to setup SWE-bench image")
                return False
            
            if not self.setup_executor():
                logger.error("Failed to setup Executor")
                return False
            
            # Core functionality tests
            self.test_basic_command_execution()
            self.test_session_management()
            self.test_check_session_functionality()
            self.test_multiple_sessions()
            self.test_error_handling()
            self.test_timeout_handling()
            
            # SWE-bench workflow test
            await self.test_swebench_workflow()
            
            # Cleanup test
            self.test_container_cleanup()
            
            # Print results
            self.print_test_results()
            
            # Return overall success
            all_passed = all(self.test_results.values())
            return all_passed
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return False
        finally:
            # Final cleanup
            if self.bash_tool:
                await self.bash_tool.close()
            if self.executor:
                self.executor.shutdown()


async def main():
    """Main test runner."""
    test_suite = ExecutorTestSuite()
    success = await test_suite.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
