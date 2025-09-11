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


async def main() -> None:
    project_root = ensure_project_root_on_path()
    print("Project root:", project_root)

    # Import tools and executor types
    from src.tools.bash_tool import BashTool
    from src.tools.edit_tool import TextEditorTool
    from src.tools.json_edit_tool import JSONEditTool
    from src.tools.search_tool import SearchTool
    from src.tools.base import ToolExecutor, ToolCall

    # Prepare workspace files
    workspace_dir = project_root / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # For edit tool target file
    target_file = workspace_dir / "test_edit_tool.txt"
    if target_file.exists():
        target_file.unlink()

    # For json edit tool file
    json_file = workspace_dir / "test_json_edit_tool.json"
    initial = {
        "config": {"version": 1, "features": {"alpha": True}},
        "users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 28}],
    }
    json_file.write_text(json.dumps(initial, indent=2, ensure_ascii=False))

    # Instantiate tools and executor
    bash_tool = BashTool()
    edit_tool = TextEditorTool()
    json_tool = JSONEditTool()
    search_tool = SearchTool()
    executor = ToolExecutor([bash_tool, edit_tool, json_tool, search_tool])

    # Stage 1: prerequisite calls that must be run before parallel batch
    # Create the text file for edit tool
    file_text = "Line A\nLine B\nLine C\n"
    create_call = ToolCall(
        name="str_replace_based_edit_tool",
        call_id="prep-edit-create",
        arguments={"command": "create", "path": str(target_file), "file_text": file_text},
    )
    prep_res = await executor.execute_tool_call(create_call)
    print("prep create:", prep_res.error or prep_res.result)

    # Stage 2: build a batch of ToolCalls and execute in parallel
    calls: list[ToolCall] = []

    # Bash: echo message (and keep session running)
    calls.append(ToolCall(name="bash", call_id="bash-hello", arguments={"command": "echo hello_parallel"}))

    # Edit tool: insert and replace, plus view
    calls.append(ToolCall(
        name="str_replace_based_edit_tool",
        call_id="edit-insert",
        arguments={"command": "insert", "path": str(target_file), "insert_line": 2, "new_str": "Inserted X\nInserted Y"},
    ))
    calls.append(ToolCall(
        name="str_replace_based_edit_tool",
        call_id="edit-replace",
        arguments={"command": "str_replace", "path": str(target_file), "old_str": "Line B\n", "new_str": "Line B (edited)\n"},
    ))
    calls.append(ToolCall(
        name="str_replace_based_edit_tool",
        call_id="edit-view",
        arguments={"command": "view", "path": str(target_file), "view_range": [1, -1]},
    ))

    # JSON tool: view/add/set/remove/view
    calls.append(ToolCall(name="json_edit_tool", call_id="json-view", arguments={"operation": "view", "file_path": str(json_file)}))
    calls.append(ToolCall(name="json_edit_tool", call_id="json-add", arguments={"operation": "add", "file_path": str(json_file), "json_path": "$.config.newKey", "value": {"enabled": True}}))
    calls.append(ToolCall(name="json_edit_tool", call_id="json-set", arguments={"operation": "set", "file_path": str(json_file), "json_path": "$.users[0].name", "value": "Alice Smith"}))
    calls.append(ToolCall(name="json_edit_tool", call_id="json-remove", arguments={"operation": "remove", "file_path": str(json_file), "json_path": "$.users[1]"}))
    calls.append(ToolCall(name="json_edit_tool", call_id="json-view-config", arguments={"operation": "view", "file_path": str(json_file), "json_path": "$.config"}))

    # Search tool: search for patterns in the workspace
    calls.append(ToolCall(name="search_tool", call_id="search-line", arguments={"pattern": "Line", "search_path": str(workspace_dir), "context_lines": 1}))
    calls.append(ToolCall(name="search_tool", call_id="search-alice", arguments={"pattern": "Alice", "search_path": str(workspace_dir), "case_insensitive": True}))
    calls.append(ToolCall(name="search_tool", call_id="search-json", arguments={"pattern": "version", "search_path": str(json_file), "file_types": "json"}))

    print("\n=== Parallel execution ===")
    results = await executor.parallel_tool_call(calls)
    for r in results:
        print(f"[{r.call_id}] {r.name} -> success={r.success}")
        if r.result:
            # Truncate long outputs for readability
            out = r.result
            print((out[:500] + "...") if len(out) > 500 else out)
        if r.error:
            print("error:", r.error)

    await executor.close_tools()


if __name__ == "__main__":
    asyncio.run(main())


