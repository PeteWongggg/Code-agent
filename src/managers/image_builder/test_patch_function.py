#!/usr/bin/env python3
"""
Simple test script to test the apply_patch_and_run_test function.
This creates a minimal test case that should work with the astropy image.
"""

from patch_and_test import apply_patch_and_run_test

def test_simple_patch():
    """Test with a simple patch that creates a new file."""
    
    # Simple patch that creates a test file
    patch_content = """diff --git a/simple_test.py b/simple_test.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/simple_test.py
@@ -0,0 +1,5 @@
+#!/usr/bin/env python3
+print("Hello from the patched file!")
+print("Python version:", __import__('sys').version)
+print("Current directory:", __import__('os').getcwd())
+print("Patch applied successfully!")
"""
    
    # Test script that runs the patched file
    test_script = """#!/bin/bash
set -e  # Exit on any error

echo "=== Starting test ==="
echo "Current directory: $(pwd)"
echo "Python version:"
python --version
echo ""

echo "=== Files in current directory ==="
ls -la
echo ""

echo "=== Checking if simple_test.py exists ==="
if [ -f "simple_test.py" ]; then
    echo "✅ simple_test.py exists"
    echo "File contents:"
    cat simple_test.py
    echo ""
    echo "=== Running the patched Python file ==="
    python simple_test.py 2>&1
    echo ""
    echo "=== Python execution completed ==="
else
    echo "❌ simple_test.py does not exist!"
    echo "Available Python files:"
    find . -name "*.py" -type f | head -10
fi

echo "=== Test completed successfully! ==="
"""
    
    # Use the actual image name
    image_name = "sweb.eval.x86_64.astropy__astropy-11693:latest"
    
    result = apply_patch_and_run_test(
        image_name=image_name,
        patch_content=patch_content,
        test_script=test_script,
        timeout=300,
    )
    
    if result['patch_apply_output']:
        print("*" * 40 + "PATCH APPLY OUTPUT:" + "*" * 40)
        print(result['patch_apply_output'])
    
    print("*" * 40 + "TEST SCRIPT OUTPUT:" + "*" * 40)
    print(result['test_output'])
    
    return result

def test_astropy_specific():
    """Test with a patch that modifies an existing astropy file."""
    
    # Patch that creates a new file (simpler approach)
    patch_content = """diff --git a/astropy_test.py b/astropy_test.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/astropy_test.py
@@ -0,0 +1,6 @@
+#!/usr/bin/env python3
+try:
+    import astropy
+    print("Astropy version:", astropy.__version__)
+    print("Astropy is available!")
+except ImportError:
+    print("Astropy not available")
+print("This file was created by a patch!")
"""
    
    # Test script that checks astropy availability
    test_script = """#!/bin/bash
set -e  # Exit on any error

echo "=== Testing Astropy Environment ==="
echo "Current directory: $(pwd)"
echo ""

echo "=== Python environment ==="
python --version
echo ""

echo "=== Files in current directory ==="
ls -la
echo ""

echo "=== Checking if astropy_test.py exists ==="
if [ -f "astropy_test.py" ]; then
    echo "✅ astropy_test.py exists"
    echo "File contents:"
    cat astropy_test.py
    echo ""
    echo "=== Running astropy test ==="
    python astropy_test.py 2>&1
    echo ""
    echo "=== Python execution completed ==="
else
    echo "❌ astropy_test.py does not exist!"
    echo "Available Python files:"
    find . -name "*.py" -type f | head -10
fi

echo "=== Test completed ==="
"""
    
    image_name = "sweb.eval.x86_64.astropy__astropy-11693:latest"
    
    print(f"\nTesting with Astropy-specific patch using image: {image_name}")
    print("=" * 60)
    
    result = apply_patch_and_run_test(
        image_name=image_name,
        patch_content=patch_content,
        test_script=test_script,
        timeout=300,
    )
    
    if result['patch_apply_output']:
        print("*" * 40 + "PATCH APPLY OUTPUT:" + "*" * 40)
        print(result['patch_apply_output'])

    print("*" * 40 + "TEST OUTPUT:" + "*" * 40)
    print(result['test_output'])
    
    return result

if __name__ == "__main__":
    print("🧪 Testing apply_patch_and_run_test function")    
    # Run simple test
    result1 = test_simple_patch()
    
    # Run astropy-specific test
    print("\n2️⃣ Running astropy-specific test...")
    result2 = test_astropy_specific()

