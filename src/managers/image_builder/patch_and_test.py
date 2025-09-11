#!/usr/bin/env python3
"""
Function to apply a patch to a built Docker image and run a test script.
Returns the bash output of running the test script after applying the patch.
"""

import time
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import docker

# Import from the installed swebench package
from swebench.harness.docker_utils import (
    cleanup_container,
    copy_to_container,
    exec_run_with_timeout,
)
from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    DOCKER_PATCH,
    DOCKER_USER,
    DOCKER_WORKDIR,
    UTF8,
)

# Git apply commands for patch application
GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]


def apply_patch_and_run_test(
    image_name: str,
    patch_content: str,
    test_script: str,
    timeout: int = 600,
    log_dir: Optional[str] = None,
    container_name_prefix: str = "patch_test"
) -> Dict[str, Any]:
    """
    Apply a patch to a Docker image and run a test script, returning the bash output.
    
    Args:
        image_name: Name of the Docker image to use
        patch_content: Content of the patch to apply (as string)
        test_script: Bash script content to run after applying the patch
        timeout: Timeout for test execution in seconds
        log_dir: Directory to save logs (optional, uses temp dir if None)
        container_name_prefix: Prefix for container name
        
    Returns:
        Dictionary containing:
        - 'success': bool - Whether the operation was successful
        - 'patch_applied': bool - Whether the patch was applied successfully
        - 'test_output': str - Bash output from running the test script
        - 'patch_apply_output': str - Output from patch application
        - 'error': str - Error message if any
        - 'container_name': str - Name of the created container
        - 'log_dir': str - Directory where logs were saved
    """
    result = {
        "success": False,
        "patch_applied": False,
        "test_output": "",
        "patch_apply_output": "",
        "error": None,
        "container_name": "",
        "log_dir": "",
    }
    
    # Set up Docker client
    client = docker.from_env()
    
    # Set up log directory
    if log_dir is None:
        log_dir = Path(tempfile.mkdtemp(prefix="patch_test_"))
    else:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
    
    result["log_dir"] = str(log_dir)
    
    # Create unique container name
    container_name = f"{container_name_prefix}_{int(time.time())}"
    result["container_name"] = container_name
    
    try:
        # Check if image exists
        try:
            client.images.get(image_name)
        except docker.errors.ImageNotFound:
            raise ValueError(f"Image {image_name} not found")
        
        # Create container from the image
        container = client.containers.create(
            image_name,
            name=container_name,
            working_dir=DOCKER_WORKDIR,
            user=DOCKER_USER,
            detach=True,
            command="tail -f /dev/null",  # Keep container running
            platform="linux/amd64",  # Force AMD64 platform for compatibility
        )
        
        try:
            # Start the container
            container.start()
            
            # Write patch to file and copy to container
            patch_file = log_dir / "patch.diff"
            patch_file.write_text(patch_content, encoding=UTF8)
            copy_to_container(container, patch_file, Path(DOCKER_PATCH))
            
            # Apply the patch to the container
            applied_patch = False
            patch_apply_output = ""
            last_unsuccessful_output = ""
            
            for git_apply_cmd in GIT_APPLY_CMDS:
                try:
                    # For the patch command, we need to handle it differently
                    if git_apply_cmd.startswith("patch"):
                        # The patch command needs the file path as the last argument
                        full_cmd = f"{git_apply_cmd} {DOCKER_PATCH}"
                    else:
                        # Git apply commands
                        full_cmd = f"{git_apply_cmd} {DOCKER_PATCH}"
                    
                    exit_code, output = container.exec_run(
                        full_cmd,
                        workdir=DOCKER_WORKDIR,
                        user=DOCKER_USER,
                    )
                    
                    output_str = output.decode(UTF8)
                    
                    if exit_code == 0:
                        # First successful run - capture and use this output
                        applied_patch = True
                        patch_apply_output = f"Command: {full_cmd}\n"
                        patch_apply_output += f"Output: {output_str}"
                        break
                    else:
                        # Unsuccessful run - store for potential use if no success
                        print(f"❌ Failed to apply patch with {git_apply_cmd}")
                        print(f"   Output: {output_str}")
                        last_unsuccessful_output = f"Command: {full_cmd}\n"
                        last_unsuccessful_output += f"Output: {output_str}"
                        
                except Exception as e:
                    error_msg = f"Error applying patch with {git_apply_cmd}: {str(e)}"
                    print(error_msg)
                    # Store exception output for potential use if no success
                    last_unsuccessful_output = f"Command: {full_cmd}\n"
                    last_unsuccessful_output += f"Error: {error_msg}"
                    continue
            
            # If no successful run, use the last unsuccessful output
            if not applied_patch:
                patch_apply_output = last_unsuccessful_output
            
            result["patch_apply_output"] = patch_apply_output
            
            if not applied_patch:
                error_msg = f"{APPLY_PATCH_FAIL}: Could not apply patch with any method"
                result["error"] = error_msg
                print(f"❌ {error_msg}")
                return result
            
            result["patch_applied"] = True
            
            # Write test script to container and run it
            test_file = log_dir / "test.sh"
            test_file.write_text(test_script, encoding=UTF8)
            copy_to_container(container, test_file, Path("/root/test.sh"))
            
            # Execute the test script with timeout
            test_output, timed_out, exec_time = exec_run_with_timeout(
                container, "/bin/bash /root/test.sh", timeout
            )
            
            result["test_output"] = test_output
            result["success"] = True
            
            if timed_out:
                print("⚠️  Test execution timed out")
            
        finally:
            # Clean up container
            cleanup_container(client, container, None)
            
    except Exception as e:
        error_msg = f"Error in apply_patch_and_run_test: {str(e)}"
        print(f"❌ {error_msg}")
        result["error"] = error_msg
    
    return result


def main():
    """Example usage of the apply_patch_and_run_test function."""
    # Simple patch that creates a test file
    example_patch = """diff --git a/test_patch.py b/test_patch.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/test_patch.py
@@ -0,0 +1,5 @@
+#!/usr/bin/env python3
+def hello():
+    return "Hello from patched file!"
+
+print(hello())
"""
    
    # Simple test script that runs the patched file
    example_test_script = """#!/bin/bash
echo "Running test script..."
echo "Current directory: $(pwd)"
echo "Files in current directory:"
ls -la
echo ""
echo "Running the patched Python file:"
python test_patch.py
echo ""
echo "Test completed successfully!"
"""
    
    # Use the actual image name you provided
    image_name = "sweb.eval.x86_64.astropy__astropy-11693:latest"
    
    result = apply_patch_and_run_test(
        image_name=image_name,
        patch_content=example_patch,
        test_script=example_test_script,
        timeout=300,
    )
    
    print(f"Patch Applied: {result['patch_applied']}")
    print(f"Error: {result['error']}")
    print(f"Container: {result['container_name']}")
    
    if result['patch_apply_output']:
        print("\nPatch Apply Output:")
        print(result['patch_apply_output'])
    
    print("\nTest Output:")
    print(result['test_output'])


if __name__ == "__main__":
    main()
