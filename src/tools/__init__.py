"""Tools module for Code Agent."""

from src.tools.base import Tool, ToolCall, ToolExecutor, ToolResult
from src.tools.bash_tool import BashTool
from src.tools.edit_tool import TextEditorTool
from src.tools.json_edit_tool import JSONEditTool

__all__ = [
    "Tool",
    "ToolResult",
    "ToolCall",
    "ToolExecutor",
    "BashTool",
    "TextEditorTool",
    "JSONEditTool",
]

tools_registry: dict[str, type[Tool]] = {
    "bash": BashTool,
    "str_replace_based_edit_tool": TextEditorTool,
    "json_edit_tool": JSONEditTool,
}
