"""Simple tests for BashTool, TextEditorTool, and JSONEditTool.

Run:
  source venv/bin/activate && python src/tools/test_tool.py
"""

import asyncio
import json
import os
from pathlib import Path
import sys


def ensure_project_root_on_path() -> Path:
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[2]  # /.../Code-agent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


async def test_bash_tool() -> None:
    print("\n=== Testing BashTool ===")
    from src.tools.bash_tool import BashTool
    tool = BashTool()

    # Start session (implicitly) and run a simple command
    res = await tool.execute({"command": "echo hello_from_bash"})
    print("bash output:", res.output.strip() if res.output else res.output)
    if res.error:
        print("bash error:", res.error)

    # Restart session
    res = await tool.execute({"command": "echo after_restart", "restart": True})
    print("bash restart output:", res.output.strip() if res.output else res.output)
    if res.error:
        print("bash restart error:", res.error)

    await tool.close()


async def test_edit_tool(project_root: Path) -> None:
    print("\n=== Testing TextEditorTool (str_replace_based_edit_tool) ===")
    from src.tools.edit_tool import TextEditorTool

    # Use workspace directory for temp files
    workspace_dir = project_root / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    target_file = workspace_dir / "test_edit_tool.txt"

    # Ensure clean state
    if target_file.exists():
        target_file.unlink()

    tool = TextEditorTool()

    # 1) create
    file_text = "Line A\nLine B\nLine C\n"
    res = await tool.execute({
        "command": "create",
        "path": str(target_file),
        "file_text": file_text,
    })
    print("create:", (res.error or res.output))

    # 2) view full
    res = await tool.execute({
        "command": "view",
        "path": str(target_file),
    })
    print("view (full):\n", res.output)

    # 3) insert after line 2
    res = await tool.execute({
        "command": "insert",
        "path": str(target_file),
        "insert_line": 2,
        "new_str": "Inserted X\nInserted Y",
    })
    print("insert:", (res.error or res.output))

    # 4) str_replace (replace exact line)
    old_str = "Line B\n"
    new_str = "Line B (edited)\n"
    res = await tool.execute({
        "command": "str_replace",
        "path": str(target_file),
        "old_str": old_str,
        "new_str": new_str,
    })
    print("str_replace:", (res.error or res.output))

    # 5) view range
    res = await tool.execute({
        "command": "view",
        "path": str(target_file),
        "view_range": [1, -1],
    })
    print("view (range):\n", res.output)


async def test_json_edit_tool(project_root: Path) -> None:
    print("\n=== Testing JSONEditTool ===")
    from src.tools.json_edit_tool import JSONEditTool

    workspace_dir = project_root / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    json_file = workspace_dir / "test_json_edit_tool.json"

    # Prepare initial JSON file
    initial = {
        "config": {
            "version": 1,
            "features": {"alpha": True}
        },
        "users": [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 28}
        ]
    }
    json_file.write_text(json.dumps(initial, indent=2, ensure_ascii=False))

    tool = JSONEditTool()

    # 1) view whole
    res = await tool.execute({
        "operation": "view",
        "file_path": str(json_file),
    })
    print("view (whole):\n", res.output)

    # 2) add new key under config
    res = await tool.execute({
        "operation": "add",
        "file_path": str(json_file),
        "json_path": "$.config.newKey",
        "value": {"enabled": True},
    })
    print("add:", (res.error or res.output))

    # 3) set users[0].name
    res = await tool.execute({
        "operation": "set",
        "file_path": str(json_file),
        "json_path": "$.users[0].name",
        "value": "Alice Smith",
    })
    print("set:", (res.error or res.output))

    # 4) remove users[1]
    res = await tool.execute({
        "operation": "remove",
        "file_path": str(json_file),
        "json_path": "$.users[1]",
    })
    print("remove:", (res.error or res.output))

    # 5) view specific path
    res = await tool.execute({
        "operation": "view",
        "file_path": str(json_file),
        "json_path": "$.config",
    })
    print("view ($.config):\n", res.output)


async def main() -> None:
    project_root = ensure_project_root_on_path()
    print("Project root:", project_root)
    await test_bash_tool()
    await test_edit_tool(project_root)
    await test_json_edit_tool(project_root)


if __name__ == "__main__":
    asyncio.run(main())


