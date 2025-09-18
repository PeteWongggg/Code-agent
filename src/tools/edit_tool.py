from pathlib import Path
from typing import override
import shlex

from src.tools.base import Tool, ToolCallArguments, ToolError, ToolExecResult, ToolParameter
from src.tools.run import maybe_truncate, run
from src.tools.executor import Executor

EditToolSubCommands = [
    "view",
    "create",
    "str_replace",
    "insert",
]
SNIPPET_LINES: int = 4


class TextEditorTool(Tool):
    """Tool to replace a string in a file."""

    def __init__(self, model_provider: str | None = None, executor: Executor | None = None) -> None:
        super().__init__(model_provider)
        self.executor = executor

    @override
    def get_model_provider(self) -> str | None:
        return self._model_provider

    @override
    def get_name(self) -> str:
        return "str_replace_based_edit_tool"

    @override
    def get_description(self) -> str:
        return """Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file !!! If you know that the `path` already exists, please remove it first and then perform the `create` operation!
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
"""

    @override
    def get_parameters(self) -> list[ToolParameter]:
        """Get the parameters for the str_replace_based_edit_tool."""
        return [
            ToolParameter(
                name="command",
                type="string",
                description=f"The commands to run. Allowed options are: {', '.join(EditToolSubCommands)}.",
                required=True,
                enum=EditToolSubCommands,
            ),
            ToolParameter(
                name="file_text",
                type="string",
                description="Required parameter of `create` command, with the content of the file to be created.",
            ),
            ToolParameter(
                name="insert_line",
                type="integer",
                description="Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.",
            ),
            ToolParameter(
                name="new_str",
                type="string",
                description="Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.",
            ),
            ToolParameter(
                name="old_str",
                type="string",
                description="Required parameter of `str_replace` command containing the string in `path` to replace.",
            ),
            ToolParameter(
                name="path",
                type="string",
                description="Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.",
                required=True,
            ),
            ToolParameter(
                name="view_range",
                type="array",
                description="Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.",
                items={"type": "integer"},
            ),
        ]

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        """Execute the str_replace_editor tool."""
        command = str(arguments["command"]) if "command" in arguments else None
        if command is None:
            return ToolExecResult(
                error=f"No command provided for the {self.get_name()} tool",
                error_code=-1,
            )
        path = str(arguments["path"]) if "path" in arguments else None
        if path is None:
            return ToolExecResult(
                error=f"No path provided for the {self.get_name()} tool", error_code=-1
            )
        _path = Path(path)
        try:
            self.validate_path(command, _path)
            match command:
                case "view":
                    return await self._view_handler(arguments, _path)
                case "create":
                    return self._create_handler(arguments, _path)
                case "str_replace":
                    return self._str_replace_handler(arguments, _path)
                case "insert":
                    return self._insert_handler(arguments, _path)
                case _:
                    return ToolExecResult(
                        error=f"Unrecognized command {command}. The allowed commands for the {self.name} tool are: {', '.join(EditToolSubCommands)}",
                        error_code=-1,
                    )
        except ToolError as e:
            return ToolExecResult(error=str(e), error_code=-1)

    def validate_path(self, command: str, path: Path):
        """Validate the path for the str_replace_editor tool."""
        if not path.is_absolute():
            suggested_path = Path("/") / path
            raise ToolError(
                f"The path {path} is not an absolute path, it should start with `/`. Maybe you meant {suggested_path}?"
            )
        # Check if path exists
        if not path.exists() and command != "create":
            raise ToolError(f"The path {path} does not exist. Please provide a valid path.")
        if path.exists() and command == "create":
            raise ToolError(
                f"File already exists at: {path}. Cannot overwrite files using command `create`."
            )
        # Check if the path points to a directory
        if path.is_dir() and command != "view":
            raise ToolError(
                f"The path {path} is a directory and only the `view` command can be used on directories"
            )

    async def _view(self, path: Path, view_range: list[int] | None = None) -> ToolExecResult:
        """Implement the view command"""
        if path.is_dir():
            if view_range:
                raise ToolError(
                    "The `view_range` parameter is not allowed when `path` points to a directory."
                )

            return_code, stdout, stderr = await run(rf"find {path} -maxdepth 2 -not -path '*/\.*'")
            if not stderr:
                stdout = f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{stdout}\n"
            return ToolExecResult(error_code=return_code, output=stdout, error=stderr)

        file_content = self.read_file(path)
        init_line = 1
        if view_range:
            if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise ToolError("Invalid `view_range`. It should be a list of two integers.")
            file_lines = file_content.split("\n")
            n_lines_file = len(file_lines)
            init_line, final_line = view_range
            if init_line < 1 or init_line > n_lines_file:
                raise ToolError(
                    f"Invalid `view_range`: {view_range}. Its first element `{init_line}` should be within the range of lines of the file: {[1, n_lines_file]}"
                )
            if final_line > n_lines_file:
                raise ToolError(
                    f"Invalid `view_range`: {view_range}. Its second element `{final_line}` should be smaller than the number of lines in the file: `{n_lines_file}`"
                )
            if final_line != -1 and final_line < init_line:
                raise ToolError(
                    f"Invalid `view_range`: {view_range}. Its second element `{final_line}` should be larger or equal than its first `{init_line}`"
                )

            if final_line == -1:
                file_content = "\n".join(file_lines[init_line - 1 :])
            else:
                file_content = "\n".join(file_lines[init_line - 1 : final_line])

        return ToolExecResult(
            output=self._make_output(file_content, str(path), init_line=init_line)
        )

    def str_replace(self, path: Path, old_str: str, new_str: str | None) -> ToolExecResult:
        """Implement the str_replace command, which replaces old_str with new_str in the file content"""
        # Read the file content
        file_content = self.read_file(path).expandtabs()
        old_str = old_str.expandtabs()
        new_str = new_str.expandtabs() if new_str is not None else ""

        # Check if old_str is unique in the file
        occurrences = file_content.count(old_str)
        if occurrences == 0:
            raise ToolError(
                f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}."
            )
        elif occurrences > 1:
            file_content_lines = file_content.split("\n")
            lines = [idx + 1 for idx, line in enumerate(file_content_lines) if old_str in line]
            raise ToolError(
                f"No replacement was performed. Multiple occurrences of old_str `{old_str}` in lines {lines}. Please ensure it is unique"
            )

        # Replace old_str with new_str
        new_file_content = file_content.replace(old_str, new_str)

        # Write the new content to the file
        self.write_file(path, new_file_content)

        # Create a snippet of the edited section
        replacement_line = file_content.split(old_str)[0].count("\n")
        start_line = max(0, replacement_line - SNIPPET_LINES)
        end_line = replacement_line + SNIPPET_LINES + new_str.count("\n")
        snippet = "\n".join(new_file_content.split("\n")[start_line : end_line + 1])

        # Prepare the success message
        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(snippet, f"a snippet of {path}", start_line + 1)
        success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."

        return ToolExecResult(
            output=success_msg,
        )

    def _insert(self, path: Path, insert_line: int, new_str: str) -> ToolExecResult:
        """Implement the insert command, which inserts new_str at the specified line in the file content."""
        file_text = self.read_file(path).expandtabs()
        new_str = new_str.expandtabs()
        file_text_lines = file_text.split("\n")
        n_lines_file = len(file_text_lines)

        if insert_line < 0 or insert_line > n_lines_file:
            raise ToolError(
                f"Invalid `insert_line` parameter: {insert_line}. It should be within the range of lines of the file: {[0, n_lines_file]}"
            )

        new_str_lines = new_str.split("\n")
        new_file_text_lines = (
            file_text_lines[:insert_line] + new_str_lines + file_text_lines[insert_line:]
        )
        snippet_lines = (
            file_text_lines[max(0, insert_line - SNIPPET_LINES) : insert_line]
            + new_str_lines
            + file_text_lines[insert_line : insert_line + SNIPPET_LINES]
        )

        new_file_text = "\n".join(new_file_text_lines)
        snippet = "\n".join(snippet_lines)

        self.write_file(path, new_file_text)

        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            snippet,
            "a snippet of the edited file",
            max(1, insert_line - SNIPPET_LINES + 1),
        )
        success_msg += "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."
        return ToolExecResult(
            output=success_msg,
        )

    # Note: undo_edit method is not implemented in this version as it was removed

    def read_file(self, path: Path):
        """Read the content of a file from a given path; raise a ToolError if an error occurs."""
        try:
            return path.read_text()
        except Exception as e:
            raise ToolError(f"Ran into {e} while trying to read {path}") from None

    def write_file(self, path: Path, file: str):
        """Write the content of a file to a given path; raise a ToolError if an error occurs."""
        try:
            _ = path.write_text(file)
        except Exception as e:
            raise ToolError(f"Ran into {e} while trying to write to {path}") from None

    def container_read_file(self, path: Path, session_id: str = "0") -> str:
        """Read the content of a file from a container using cat command."""
        if not self.executor:
            raise ToolError("No executor provided for container operations")
        
        try:
            # Check if session is alive and restart if needed
            if not self.executor.check_session():
                raise ToolError("Container session is not alive and could not be restarted")
            
            # Use cat command to read file content
            command = f"cat {path}"
            return_code, output = self.executor.execute(session_id, command)
            
            if return_code != 0:
                raise ToolError(f"Failed to read file {path} from container. Exit code: {return_code}, Output: {output}")
            
            # Clean the output by removing only the command echo, preserving file content exactly
            lines = output.split('\n')
            # Remove the first line if it contains the command echo
            if lines and f"cat {path}" in lines[0]:
                lines = lines[2:-1]
            
            return '\n'.join(lines)
        except Exception as e:
            raise ToolError(f"Ran into {e} while trying to read {path} from container") from None

    def container_write_file(self, path: Path, content: str, session_id: str = "0") -> None:
        """Write content to a file in a container using cat with here document."""
        if not self.executor:
            raise ToolError("No executor provided for container operations")
        
        try:
            # Check if session is alive and restart if needed
            if not self.executor.check_session():
                raise ToolError("Container session is not alive and could not be restarted")
            
            # Use cat with here document to handle long content
            # Create a script that writes the content using here document
            script_content = f"""cat > {path} << 'EOF'
{content}
EOF"""
            
            return_code, output = self.executor.execute(session_id, script_content)
            
            if return_code != 0:
                raise ToolError(f"Failed to write to file {path} in container. Exit code: {return_code}, Output: {output}")
                
        except Exception as e:
            raise ToolError(f"Ran into {e} while trying to write to {path} in container") from None

    def container_str_replace(self, path: Path, old_str: str, new_str: str | None, session_id: str = "0") -> ToolExecResult:
        """Replace old_str with new_str in a file in a container using sed command."""
        if not self.executor:
            raise ToolError("No executor provided for container operations")
        
        try:
            # Check if session is alive and restart if needed
            if not self.executor.check_session():
                raise ToolError("Container session is not alive and could not be restarted")
            
            # First, read the file to check if old_str exists
            file_content = self.container_read_file(path, session_id)
            
            # Check if old_str is unique in the file
            occurrences = file_content.count(old_str)
            if occurrences == 0:
                raise ToolError(
                    f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}."
                )
            elif occurrences > 1:
                file_content_lines = file_content.split("\n")
                lines = [idx + 1 for idx, line in enumerate(file_content_lines) if old_str in line]
                raise ToolError(
                    f"No replacement was performed. Multiple occurrences of old_str `{old_str}` in lines {lines}. Please ensure it is unique"
                )
            
            # Escape special characters for sed
            escaped_old = self._escape_sed(old_str)
            escaped_new = self._escape_sed(new_str) if new_str is not None else ""
            
            # Check session again before sed command
            if not self.executor.check_session():
                raise ToolError("Container session is not alive and could not be restarted")
            
            # Use sed for in-place replacement
            command = f"sed -i 's/{escaped_old}/{escaped_new}/g' {path}"
            return_code, output = self.executor.execute(session_id, command)
            
            if return_code != 0:
                raise ToolError(f"Failed to replace string in file {path} in container. Exit code: {return_code}, Output: {output}")
            
            # Read the file to show a snippet of the changes
            try:
                file_content = self.container_read_file(path, session_id)
                # Create a simple snippet showing the change
                lines = file_content.split('\n')
                snippet_lines = lines[:min(10, len(lines))]  # Show first 10 lines
                snippet = '\n'.join(snippet_lines)
                
                success_msg = f"The file {path} has been edited in container. "
                success_msg += self._make_output(snippet, f"a snippet of {path}", init_line=1)
                success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."
                
                return ToolExecResult(output=success_msg)
            except Exception:
                # If we can't read the file for snippet, just return success
                return ToolExecResult(output=f"Successfully replaced string in file {path} in container.")
                
        except Exception as e:
            raise ToolError(f"Ran into {e} while trying to replace string in {path} in container") from None

    def _escape_sed(self, text: str) -> str:
        """Escape special characters in text for use with sed command."""
        # Escape sed special characters: / \ & 
        escaped = text.replace('\\', '\\\\')  # Escape backslashes first
        escaped = escaped.replace('/', '\\/')  # Escape forward slashes
        escaped = escaped.replace('&', '\\&')  # Escape ampersands
        escaped = escaped.replace('\n', '\\n')  # Handle newlines
        escaped = escaped.replace('\t', '\\t')  # Handle tabs
        return escaped

    def _make_output(
        self,
        file_content: str,
        file_descriptor: str,
        init_line: int = 1,
        expand_tabs: bool = True,
    ):
        """Generate output for the CLI based on the content of a file."""
        file_content = maybe_truncate(file_content)
        if expand_tabs:
            file_content = file_content.expandtabs()
        file_content = "\n".join(
            [f"{i + init_line:6}\t{line}" for i, line in enumerate(file_content.split("\n"))]
        )
        return (
            f"Here's the result of running `cat -n` on {file_descriptor}:\n" + file_content + "\n"
        )

    async def _view_handler(self, arguments: ToolCallArguments, _path: Path) -> ToolExecResult:
        view_range = arguments.get("view_range", None)
        if view_range is None:
            return await self._view(_path, None)
        if not (isinstance(view_range, list) and all(isinstance(i, int) for i in view_range)):
            return ToolExecResult(
                error="Parameter `view_range` should be a list of integers.",
                error_code=-1,
            )
        view_range_int: list[int] = [i for i in view_range if isinstance(i, int)]
        return await self._view(_path, view_range_int)

    def _create_handler(self, arguments: ToolCallArguments, _path: Path) -> ToolExecResult:
        file_text = arguments.get("file_text", None)
        if not isinstance(file_text, str):
            return ToolExecResult(
                error="Parameter `file_text` is required and must be a string for command: create",
                error_code=-1,
            )
        self.write_file(_path, file_text)
        return ToolExecResult(output=f"File created successfully at: {_path}")

    def _str_replace_handler(self, arguments: ToolCallArguments, _path: Path) -> ToolExecResult:
        old_str = arguments.get("old_str") if "old_str" in arguments else None
        if not isinstance(old_str, str):
            return ToolExecResult(
                error="Parameter `old_str` is required and should be a string for command: str_replace",
                error_code=-1,
            )
        new_str = arguments.get("new_str") if "new_str" in arguments else None
        if not (new_str is None or isinstance(new_str, str)):
            return ToolExecResult(
                error="Parameter `new_str` should be a string or null for command: str_replace",
                error_code=-1,
            )
        return self.str_replace(_path, old_str, new_str)

    def _insert_handler(self, arguments: ToolCallArguments, _path: Path) -> ToolExecResult:
        insert_line = arguments.get("insert_line") if "insert_line" in arguments else None
        if not isinstance(insert_line, int):
            return ToolExecResult(
                error="Parameter `insert_line` is required and should be integer for command: insert",
                error_code=-1,
            )
        new_str_to_insert = arguments.get("new_str") if "new_str" in arguments else None
        if not isinstance(new_str_to_insert, str):
            return ToolExecResult(
                error="Parameter `new_str` is required for command: insert",
                error_code=-1,
            )
        return self._insert(_path, insert_line, new_str_to_insert)
