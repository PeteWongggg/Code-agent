import asyncio
import sys
import os
from pathlib import Path
from typing import Optional
import logging
import tempfile
import shutil

# Add the necessary paths to sys.path
sys.path.append('/Users/hengzhi/Code-agent')
sys.path.append('/Users/hengzhi/Code-agent/src')

# Import required modules
from src.tools.search_tool import SearchTool
from src.tools.executor import Executor
from src.managers.image_builder.build_image import SWEBenchImageBuilder
from src.tools.base import ToolCallArguments
from src.tools.edit_tool import TextEditorTool

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


class SearchToolTest:    
    def __init__(self, workspace_dir: str = "/tmp/search_tool_test"):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.executor: Optional[Executor] = None
        self.search_tool: Optional[SearchTool] = None
        self.edit_tool: Optional[TextEditorTool] = None
        self.session_id: Optional[str] = None
        
        self.image_builder: Optional[SWEBenchImageBuilder] = None
        self.current_instance: Optional[SWEbenchInstance] = None
        self.image_name: Optional[str] = None
        
        self.test_results = {}
        self.test_files_created = []
        
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
    
    def setup_executor_and_search_tool(self) -> bool:
        """Set up executor and search tool"""
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
            
            # Verify ripgrep is available
            logger.info("Checking for ripgrep availability...")
            return_code, output = self.executor.execute(self.session_id, "which rg")
            if return_code != 0:
                logger.error("Ripgrep not available in container. Please rebuild images with ripgrep installed.")
                return False
            else:
                logger.info("✓ Ripgrep is available")
            
            self.search_tool = SearchTool(model_provider="openai", executor=self.executor)
            self.edit_tool = TextEditorTool(model_provider="openai", executor=self.executor)
            logger.info("✓ Executor and SearchTool setup completed")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to setup executor and search tool: {e}")
            return False
    
    def create_test_files(self) -> bool:
        """Create test files for search operations"""
        try:
            logger.info("Creating test files for search operations...")
            
            # Create test directory structure
            test_dir = self.workspace_dir / "test_search"
            test_dir.mkdir(exist_ok=True)
            
            # Create Python test file
            python_file = test_dir / "test_script.py"
            python_content = """#!/usr/bin/env python3
# Test script for search functionality
import os
import sys
from pathlib import Path

class TestClass:
    def __init__(self, name: str):
        self.name = name
        self.value = 42
    
    def get_name(self):
        return self.name
    
    def calculate(self, x: int, y: int) -> int:
        result = x + y
        return result * self.value

def main():
    print("Hello, World!")
    test_obj = TestClass("test_instance")
    result = test_obj.calculate(10, 20)
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
"""
            python_file.write_text(python_content)
            self.test_files_created.append(python_file)
            
            # Create JavaScript test file
            js_file = test_dir / "test_script.js"
            js_content = """// JavaScript test file for search functionality
const fs = require('fs');
const path = require('path');

class TestClass {
    constructor(name) {
        this.name = name;
        this.value = 42;
    }
    
    getName() {
        return this.name;
    }
    
    calculate(x, y) {
        const result = x + y;
        return result * this.value;
    }
}

function main() {
    console.log("Hello, World!");
    const testObj = new TestClass("test_instance");
    const result = testObj.calculate(10, 20);
    console.log(`Result: ${result}`);
}

if (require.main === module) {
    main();
}
"""
            js_file.write_text(js_content)
            self.test_files_created.append(js_file)
            
            # Create Markdown test file
            md_file = test_dir / "README.md"
            md_content = """# Test Search Functionality

This is a test markdown file for search functionality.

## Features

- Search for text patterns
- Support for multiple file types
- Context-aware results
- Regex pattern matching

## Usage

```python
# Example Python code
def search_function(pattern, path):
    return find_matches(pattern, path)
```

## Notes

- This file contains various text patterns
- It's used for testing search capabilities
- Multiple languages are supported
"""
            md_file.write_text(md_content)
            self.test_files_created.append(md_file)
            
            # Create nested directory with more files
            nested_dir = test_dir / "nested"
            nested_dir.mkdir(exist_ok=True)
            
            nested_file = nested_dir / "nested_test.py"
            nested_content = """# Nested test file
def nested_function():
    return "This is a nested function"

class NestedClass:
    def method(self):
        return "nested method"
"""
            nested_file.write_text(nested_content)
            self.test_files_created.append(nested_file)
            
            logger.info(f"✓ Created {len(self.test_files_created)} test files")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to create test files: {e}")
            return False
    
    
    async def test_container_search_basic(self) -> bool:
        """Test basic container search functionality"""
        try:
            logger.info("Testing container basic search...")
            
            # Copy test files to container
            test_dir_path = self.workspace_dir / "test_search"
            container_test_dir = "/testbed/search_test"
            
            # Create directory in container
            return_code, _ = self.executor.execute(self.session_id, f"mkdir -p {container_test_dir}")
            if return_code != 0:
                logger.error("Failed to create test directory in container")
                self.test_results["container_basic_search"] = False
                return False
            
            # Copy files to container (simplified - in real scenario would use docker cp)
            # For this test, we'll create files directly in container
            python_content = """#!/usr/bin/env python3
# Test script for search functionality
import os
import sys
from pathlib import Path

class TestClass:
    def __init__(self, name: str):
        self.name = name
        self.value = 42
    
    def get_name(self):
        return self.name
    
    def calculate(self, x: int, y: int) -> int:
        result = x + y
        return result * self.value

def main():
    print("Hello, World!")
    test_obj = TestClass("test_instance")
    result = test_obj.calculate(10, 20)
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
"""
            
            # Write test file to container using edit tool
            try:
                self.edit_tool.container_write_file(
                    Path(f"{container_test_dir}/test_script.py"), 
                    python_content, 
                    self.session_id
                )
                logger.info("✓ Test file created successfully in container")
            except Exception as e:
                logger.error(f"Failed to create test file in container: {e}")
                self.test_results["container_basic_search"] = False
                return False
            
            
            # Test container search
            args = ToolCallArguments({
                "pattern": "def main",
                "search_path": container_test_dir,
                "context_lines": 1,
                "max_results": 10
            })
            
            result = self.search_tool.container_search(args, self.session_id)
            
            if result.error:
                logger.error(f"✗ Container basic search failed: {result.error}")
                self.test_results["container_basic_search"] = False
                return False
            
            if "def main" in result.output:
                logger.info("✓ Container basic search successful")
                self.test_results["container_basic_search"] = True
                return True
            else:
                logger.error(f"✗ Container basic search failed - pattern not found. Output: {result.output}")
                self.test_results["container_basic_search"] = False
                return False
                
        except Exception as e:
            logger.error(f"✗ Container basic search test failed: {e}")
            self.test_results["container_basic_search"] = False
            return False
    
    async def test_container_search_regex(self) -> bool:
        """Test container regex search functionality"""
        try:
            logger.info("Testing container regex search...")
            
            container_test_dir = "/testbed/search_test"
            
            # Test regex pattern search in container
            args = ToolCallArguments({
                "pattern": r"class\s+\w+",
                "search_path": container_test_dir,
                "context_lines": 1,
                "max_results": 10
            })
            
            result = self.search_tool.container_search(args, self.session_id)
            
            if result.error:
                logger.error(f"✗ Container regex search failed: {result.error}")
                self.test_results["container_regex_search"] = False
                return False
            
            if "class" in result.output and "TestClass" in result.output:
                logger.info("✓ Container regex search successful")
                self.test_results["container_regex_search"] = True
                return True
            else:
                logger.error(f"✗ Container regex search failed - pattern not found. Output: {result.output}")
                self.test_results["container_regex_search"] = False
                return False
                
        except Exception as e:
            logger.error(f"✗ Container regex search test failed: {e}")
            self.test_results["container_regex_search"] = False
            return False
    
    async def test_container_search_case_insensitive(self) -> bool:
        """Test container case-insensitive search"""
        try:
            logger.info("Testing container case-insensitive search...")
            
            container_test_dir = "/testbed/search_test"
            
            # Test case-insensitive search
            args = ToolCallArguments({
                "pattern": "HELLO",
                "search_path": container_test_dir,
                "case_insensitive": True,
                "max_results": 10
            })
            
            result = self.search_tool.container_search(args, self.session_id)
            
            if result.error:
                logger.error(f"✗ Container case-insensitive search failed: {result.error}")
                self.test_results["container_case_insensitive"] = False
                return False
            
            if "Hello" in result.output or "HELLO" in result.output:
                logger.info("✓ Container case-insensitive search successful")
                self.test_results["container_case_insensitive"] = True
                return True
            else:
                logger.error(f"✗ Container case-insensitive search failed - pattern not found. Output: {result.output}")
                self.test_results["container_case_insensitive"] = False
                return False
                
        except Exception as e:
            logger.error(f"✗ Container case-insensitive search test failed: {e}")
            self.test_results["container_case_insensitive"] = False
            return False
    
    async def test_container_search_context_lines(self) -> bool:
        """Test container search with context lines"""
        try:
            logger.info("Testing container search with context lines...")
            
            container_test_dir = "/testbed/search_test"
            
            # Test search with context lines
            args = ToolCallArguments({
                "pattern": "def calculate",
                "search_path": container_test_dir,
                "context_lines": 3,
                "max_results": 10
            })
            
            result = self.search_tool.container_search(args, self.session_id)
            
            if result.error:
                logger.error(f"✗ Container context search failed: {result.error}")
                self.test_results["container_context_search"] = False
                return False
            
            # Check if context lines are present (should show lines before/after the match)
            lines = result.output.split('\n')
            context_found = any('  ' in line and ':' in line for line in lines)  # Context lines have different formatting
            
            if "def calculate" in result.output and context_found:
                logger.info("✓ Container context search successful")
                self.test_results["container_context_search"] = True
                return True
            else:
                logger.error(f"✗ Container context search failed - context not found. Output: {result.output}")
                self.test_results["container_context_search"] = False
                return False
                
        except Exception as e:
            logger.error(f"✗ Container context search test failed: {e}")
            self.test_results["container_context_search"] = False
            return False
    
    async def test_container_search_no_matches(self) -> bool:
        """Test container search when no matches are found"""
        try:
            logger.info("Testing container search with no matches...")
            
            container_test_dir = "/testbed/search_test"
            
            # Test search for non-existent pattern
            args = ToolCallArguments({
                "pattern": "nonexistent_pattern_xyz",
                "search_path": container_test_dir,
                "max_results": 10
            })
            
            result = self.search_tool.container_search(args, self.session_id)
            
            if result.error:
                logger.error(f"✗ Container no-matches search failed: {result.error}")
                self.test_results["container_no_matches"] = False
                return False
            
            if "No matches found" in result.output:
                logger.info("✓ Container no-matches search successful")
                self.test_results["container_no_matches"] = True
                return True
            else:
                logger.error(f"✗ Container no-matches search failed - should have no matches. Output: {result.output}")
                self.test_results["container_no_matches"] = False
                return False
                
        except Exception as e:
            logger.error(f"✗ Container no-matches search test failed: {e}")
            self.test_results["container_no_matches"] = False
            return False
    
    async def test_container_search_file_types(self) -> bool:
        """Test container search with file type filtering"""
        try:
            logger.info("Testing container file type filtering...")
            
            # Create additional test files in container
            container_test_dir = "/testbed/search_test"
            
            # Create a JavaScript file
            js_content = """// JavaScript test file
const fs = require('fs');
const path = require('path');

function testFunction() {
    console.log("Hello from JS");
    return "test result";
}

class TestClass {
    constructor() {
        this.value = 42;
    }
}
"""
            
            # Create the JavaScript file using edit tool
            try:
                self.edit_tool.container_write_file(
                    Path(f"{container_test_dir}/test_script.js"), 
                    js_content, 
                    self.session_id
                )
                logger.info("✓ JS test file created successfully in container")
            except Exception as e:
                logger.error(f"Failed to create JS test file in container: {e}")
                self.test_results["container_file_type_search"] = False
                return False
            
            # Test Python files only
            args = ToolCallArguments({
                "pattern": "import",
                "search_path": container_test_dir,
                "file_types": "py",
                "max_results": 10
            })
            
            result = self.search_tool.container_search(args, self.session_id)
            
            if result.error:
                logger.error(f"✗ Container file type search failed: {result.error}")
                self.test_results["container_file_type_search"] = False
                return False
            
            if "import" in result.output and ".py" in result.output and ".js" not in result.output:
                logger.info("✓ Container file type search successful")
                self.test_results["container_file_type_search"] = True
                return True
            else:
                logger.error(f"✗ Container file type search failed. Output: {result.output}")
                self.test_results["container_file_type_search"] = False
                return False
                
        except Exception as e:
            logger.error(f"✗ Container file type search test failed: {e}")
            self.test_results["container_file_type_search"] = False
            return False
    
    async def test_container_search_error_handling(self) -> bool:
        """Test container search error handling"""
        try:
            logger.info("Testing container search error handling...")
            
            # Test search in non-existent directory
            args = ToolCallArguments({
                "pattern": "test",
                "search_path": "/nonexistent/directory",
                "max_results": 10
            })
            
            result = self.search_tool.container_search(args, self.session_id)
            
            # Should handle the error gracefully
            if result.error or "No matches found" in result.output:
                logger.info("✓ Container error handling successful")
                self.test_results["container_error_handling"] = True
                return True
            else:
                logger.error(f"✗ Container error handling failed - should have handled error. Output: {result.output}")
                self.test_results["container_error_handling"] = False
                return False
                
        except Exception as e:
            logger.error(f"✗ Container error handling test failed: {e}")
            self.test_results["container_error_handling"] = False
            return False
    
    def print_test_results(self):
        """Print comprehensive test results"""
        print(f"\n" + "="*80)
        print("SEARCH TOOL CONTAINER OPERATIONS TEST RESULTS")
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
            print("🎉 All tests passed! SearchTool works correctly with Executor.")
        else:
            print(f"⚠️  {total_tests - passed_tests} test(s) failed. Check the logs above for details.")
    
    async def cleanup(self):
        """Clean up resources"""
        if self.session_id and self.executor:
            self.executor.close_session(self.session_id)
        if self.executor:
            self.executor.shutdown()
        
        # Clean up test files
        try:
            if self.workspace_dir.exists():
                shutil.rmtree(self.workspace_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up workspace: {e}")
    
    async def run_all_tests(self) -> bool:
        """Run all search tool tests"""
        try:
            logger.info("Starting SearchTool container operations tests...")
            
            # Setup
            if not await self.setup_swebench_image():
                return False
            
            if not self.setup_executor_and_search_tool():
                return False
            
            if not self.create_test_files():
                return False
            
            # Run tests
            all_passed = True
            
            # Container tests
            try:
                result = await self.test_container_search_basic()
                if not result:
                    all_passed = False
            except Exception as e:
                logger.error(f"✗ Container basic search test failed with exception: {e}")
                all_passed = False
            
            try:
                result = await self.test_container_search_regex()
                if not result:
                    all_passed = False
            except Exception as e:
                logger.error(f"✗ Container regex search test failed with exception: {e}")
                all_passed = False
            
            try:
                result = await self.test_container_search_case_insensitive()
                if not result:
                    all_passed = False
            except Exception as e:
                logger.error(f"✗ Container case-insensitive search test failed with exception: {e}")
                all_passed = False
            
            try:
                result = await self.test_container_search_context_lines()
                if not result:
                    all_passed = False
            except Exception as e:
                logger.error(f"✗ Container context search test failed with exception: {e}")
                all_passed = False
            
            try:
                result = await self.test_container_search_no_matches()
                if not result:
                    all_passed = False
            except Exception as e:
                logger.error(f"✗ Container no-matches search test failed with exception: {e}")
                all_passed = False
            
            try:
                result = await self.test_container_search_file_types()
                if not result:
                    all_passed = False
            except Exception as e:
                logger.error(f"✗ Container file type search test failed with exception: {e}")
                all_passed = False
            
            try:
                result = await self.test_container_search_error_handling()
                if not result:
                    all_passed = False
            except Exception as e:
                logger.error(f"✗ Container error handling test failed with exception: {e}")
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
    test = SearchToolTest()    
    success = await test.run_all_tests()
    
    if success:
        print("\n🎉 All SearchTool container operations tests completed successfully!")
    else:
        print("\n⚠️  Some SearchTool container operations tests failed. Check the output above.")


if __name__ == "__main__":
    asyncio.run(main())
