"""Search tool for finding files based on text content using ripgrep (rg)."""

import asyncio
import json
import re
from pathlib import Path
from typing import override

from src.tools.base import Tool, ToolCallArguments, ToolError, ToolExecResult, ToolParameter
from src.tools.run import run
from src.tools.executor import Executor


class SearchTool(Tool):
    """Tool for searching files based on text content using ripgrep."""

    def __init__(self, model_provider: str | None = None, executor: Executor | None = None) -> None:
        super().__init__(model_provider)
        self._executor = executor

    @override
    def get_model_provider(self) -> str | None:
        return self._model_provider

    @override
    def get_name(self) -> str:
        return "search_tool"

    @override
    def get_description(self) -> str:
        return """Search tool for finding files based on text content using ripgrep (rg)
* Searches for text patterns in files and directories recursively
* Returns file paths, line numbers, and surrounding context
* Supports regex patterns and various search options
* Fast and efficient search using ripgrep engine

Features:
- Pattern matching with regex support
- Line number display
- Context lines (before/after matches)
- File type filtering
- Case sensitivity control
- Hidden file inclusion
- Binary file handling

Example patterns:
- Simple text: "function main"
- Regex: "def\\s+\\w+\\s*\\("
- Case insensitive: "TODO" (with case_insensitive=true)
"""

    @override
    def get_parameters(self) -> list[ToolParameter]:
        """Get the parameters for the search tool."""
        params = [
            ToolParameter(
                name="pattern",
                type="string",
                description="The search pattern (regex supported). Can be plain text or regex pattern.",
                required=True,
            ),
            ToolParameter(
                name="search_path",
                type="string",
                description="The directory or file path to search in. Must be an absolute path.",
                required=True,
            ),
            ToolParameter(
                type="integer",
                name="context_lines",
                description="Number of context lines to show before and after each match. Default: 2.",
                required=False,
            ),
            ToolParameter(
                type="boolean",
                name="case_insensitive",
                description="Whether to perform case-insensitive search. Default: false.",
                required=False,
            ),
            ToolParameter(
                type="boolean",
                name="include_hidden",
                description="Whether to include hidden files and directories. Default: false.",
                required=False,
            ),
            ToolParameter(
                type="boolean",
                name="include_binary",
                description="Whether to search in binary files. Default: false.",
                required=False,
            ),
            ToolParameter(
                type="string",
                name="file_types",
                description="Comma-separated list of file types to search (e.g., 'py,js,md'). Optional.",
                required=False,
            ),
            ToolParameter(
                type="integer",
                name="max_results",
                description="Maximum number of results to return. Default: 100.",
                required=False,
            ),
        ]
        
        # Add container-specific parameters if executor is available
        if self._executor:
            params.extend([
                ToolParameter(
                    name="session_id",
                    type="string",
                    description="Session ID for container execution. Required when using container executor.",
                    required=True,
                ),
                ToolParameter(
                    name="use_container",
                    type="boolean",
                    description="Whether to execute search in container. Default: true when executor is provided.",
                    required=False,
                ),
            ])
        
        return params

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        """Execute the search operation."""
        try:
            pattern = str(arguments.get("pattern", ""))
            if not pattern:
                return ToolExecResult(error="Pattern parameter is required", error_code=-1)

            search_path_str = str(arguments.get("search_path", ""))
            if not search_path_str:
                return ToolExecResult(error="search_path parameter is required", error_code=-1)

            search_path = Path(search_path_str)
            if not search_path.is_absolute():
                return ToolExecResult(
                    error=f"Search path must be absolute: {search_path}", error_code=-1
                )

            if not search_path.exists():
                return ToolExecResult(
                    error=f"Search path does not exist: {search_path}", error_code=-1
                )

            # Parse optional parameters
            context_lines = int(arguments.get("context_lines", 2))
            case_insensitive = bool(arguments.get("case_insensitive", False))
            include_hidden = bool(arguments.get("include_hidden", False))
            include_binary = bool(arguments.get("include_binary", False))
            file_types = arguments.get("file_types")
            max_results = int(arguments.get("max_results", 100))

            # Build ripgrep command
            cmd_parts = ["rg"]

            # Add context lines
            if context_lines > 0:
                cmd_parts.extend(["-C", str(context_lines)])

            # Add case sensitivity
            if case_insensitive:
                cmd_parts.append("-i")

            # Add hidden files
            if include_hidden:
                cmd_parts.append("--hidden")

            # Add binary files
            if include_binary:
                cmd_parts.append("--binary")
            else:
                cmd_parts.append("--no-binary")

            # Add file types
            if file_types and isinstance(file_types, str):
                for file_type in file_types.split(","):
                    file_type = file_type.strip()
                    if file_type:
                        cmd_parts.extend(["-t", file_type])

            # Add line numbers and filename
            cmd_parts.extend(["-n", "-H"])

            # Add max results (approximate by limiting to max_results * 2 to account for context)
            cmd_parts.extend(["-m", str(max_results * 2)])

            # Add pattern and search path (quote pattern to handle spaces)
            cmd_parts.extend([f'"{pattern}"', str(search_path)])

            # Execute the command
            return_code, stdout, stderr = await run(" ".join(cmd_parts))

            if return_code == 0:
                # Parse and format results
                results = self._parse_rg_output(stdout)
                formatted_output = self._format_results(results, max_results)
                return ToolExecResult(output=formatted_output)
            elif return_code == 1:
                # No matches found
                return ToolExecResult(output=f"No matches found for pattern: {pattern}")
            else:
                # Error occurred
                error_msg = stderr if stderr else f"ripgrep exited with code {return_code}"
                return ToolExecResult(error=error_msg, error_code=return_code)

        except Exception as e:
            return ToolExecResult(error=f"Search tool error: {str(e)}", error_code=-1)


    def container_search(self, arguments: ToolCallArguments, session_id: str = "0") -> ToolExecResult:
        """在容器中执行搜索操作"""
        if not self._executor:
            return ToolExecResult(error="No executor provided for container search", error_code=-1)
        
        try:
            # 从参数中提取搜索相关参数
            pattern = str(arguments.get("pattern", ""))
            if not pattern:
                return ToolExecResult(error="Pattern parameter is required", error_code=-1)

            search_path_str = str(arguments.get("search_path", ""))
            if not search_path_str:
                return ToolExecResult(error="search_path parameter is required", error_code=-1)

            # 解析可选参数
            context_lines = int(arguments.get("context_lines", 2))
            case_insensitive = bool(arguments.get("case_insensitive", False))
            include_hidden = bool(arguments.get("include_hidden", False))
            include_binary = bool(arguments.get("include_binary", False))
            file_types = arguments.get("file_types")
            max_results = int(arguments.get("max_results", 100))

            # 构建 ripgrep 命令
            cmd_parts = ["rg"]

            # 添加上下文行数
            if context_lines > 0:
                cmd_parts.extend(["-C", str(context_lines)])

            # 添加大小写敏感性
            if case_insensitive:
                cmd_parts.append("-i")

            # 添加隐藏文件
            if include_hidden:
                cmd_parts.append("--hidden")

            # 添加二进制文件
            if include_binary:
                cmd_parts.append("--binary")
            else:
                cmd_parts.append("--no-binary")

            # 添加文件类型
            if file_types and isinstance(file_types, str):
                for file_type in file_types.split(","):
                    file_type = file_type.strip()
                    if file_type:
                        cmd_parts.extend(["-t", file_type])

            # 添加行号和文件名
            cmd_parts.extend(["-n", "-H"])

            # 添加最大结果数
            cmd_parts.extend(["-m", str(max_results * 2)])

            # 添加搜索模式和路径
            cmd_parts.extend([f'"{pattern}"', search_path_str])

            # 在容器中执行命令
            command = " ".join(cmd_parts)
            return_code, output = self._executor.execute(session_id, command)

            if return_code == 0:
                # 解析和格式化结果
                results = self._parse_rg_output(output)
                formatted_output = self._format_results(results, max_results)
                return ToolExecResult(output=formatted_output)
            elif return_code == 1:
                # 没有找到匹配
                return ToolExecResult(output=f"No matches found for pattern: {pattern}")
            else:
                # 发生错误
                return ToolExecResult(error=f"ripgrep exited with code {return_code}. Output: {output}", error_code=return_code)

        except Exception as e:
            return ToolExecResult(error=f"Container search error: {str(e)}", error_code=-1)

    def _parse_rg_output(self, output: str) -> list[dict]:
        """Parse ripgrep output into structured results."""
        results = []
        current_file = None

        for line in output.split("\n"):
            if not line.strip():
                continue

            # Parse ripgrep output format: file:line:content or file:line-content
            if ":" in line:
                # Check if this line contains a match (has line number)
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    file_path = parts[0].strip()
                    line_info = parts[1].strip()
                    content = parts[2].strip()
                    
                    # Check if line_info is a number (match line) or contains dash (context line)
                    if line_info.isdigit():
                        # This is a match line
                        line_num = int(line_info)
                        current_file = file_path
                        results.append({
                            "file": file_path,
                            "line": line_num,
                            "content": content,
                            "full_line": line,
                            "is_match": True
                        })
                    elif "-" in line_info and current_file == file_path:
                        # This is a context line (before/after match)
                        # Extract line number from context line format like "12-15" or "12-"
                        try:
                            if "-" in line_info:
                                line_num = int(line_info.split("-")[0])
                            else:
                                continue
                            
                            results.append({
                                "file": file_path,
                                "line": line_num,
                                "content": content,
                                "full_line": line,
                                "is_match": False
                            })
                        except ValueError:
                            continue

        return results

    def _format_results(self, results: list[dict], max_results: int) -> str:
        """Format search results for display."""
        if not results:
            return "No matches found."

        # Filter only match lines for counting
        match_results = [r for r in results if r.get("is_match", True)]
        limited_results = results[:max_results]
        
        output_lines = [f"Found {len(match_results)} matches:"]
        output_lines.append("=" * 50)

        current_file = None
        for result in limited_results:
            file_path = result["file"]
            line_num = result["line"]
            content = result["content"]
            is_match = result.get("is_match", True)

            # Add file header if this is a new file
            if current_file != file_path:
                current_file = file_path
                output_lines.append(f"\n📁 {file_path}")
                output_lines.append("-" * (len(file_path) + 4))

            # Add line with appropriate prefix
            prefix = "  " if is_match else "  "  # Match lines get no special prefix
            marker = "▶" if is_match else "  "  # Mark actual matches
            output_lines.append(f"{marker} {line_num:4d}: {content}")

        if len(results) > max_results:
            output_lines.append(f"\n... and {len(results) - max_results} more lines")

        return "\n".join(output_lines)
