import asyncio
import os
from typing import override

from src.tools.base import Tool, ToolCallArguments, ToolError, ToolExecResult, ToolParameter
from src.tools.executor import Executor


class _BashSession:
    """A session of a bash shell."""

    _started: bool
    _timed_out: bool

    command: str = "/bin/bash"
    _output_delay: float = 0.2  # seconds
    _timeout: float = 120.0  # seconds
    _sentinel: str = ",,,,bash-command-exit-__ERROR_CODE__-banner,,,,"  # `__ERROR_CODE__` will be replaced by `$?` or `!errorlevel!` later

    def __init__(self) -> None:
        self._started = False
        self._timed_out = False
        self._process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        if self._started:
            return

        # Windows compatibility: os.setsid not available

        if os.name != "nt":  # Unix-like systems
            self._process = await asyncio.create_subprocess_shell(
                self.command,
                shell=True,
                bufsize=0,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=os.setsid,
            )
        else:
            self._process = await asyncio.create_subprocess_shell(
                "cmd.exe /v:on",  # enable delayed expansion to allow `echo !errorlevel!`
                shell=True,
                bufsize=0,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

        self._started = True

    async def stop(self) -> None:
        """Terminate the bash shell."""
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process is None:
            return
        if self._process.returncode is not None:
            return
        self._process.terminate()

        # Wait until the process has truly terminated.
        stdout, stderr = await self._process.communicate()

    async def run(self, command: str) -> ToolExecResult:
        """Execute a command in the bash shell."""
        if not self._started or self._process is None:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return ToolExecResult(
                error=f"bash has exited with returncode {self._process.returncode}. tool must be restarted.",
                error_code=-1,
            )
        if self._timed_out:
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            )

        # we know these are not None because we created the process with PIPEs
        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        error_code = 0

        sentinel_before, pivot, sentinel_after = self._sentinel.partition("__ERROR_CODE__")
        assert pivot == "__ERROR_CODE__"

        errcode_retriever = "!errorlevel!" if os.name == "nt" else "$?"
        command_sep = "&" if os.name == "nt" else ";"

        # send command to the process
        self._process.stdin.write(
            b"(\n"
            + command.encode()
            + f"\n){command_sep} echo {self._sentinel.replace('__ERROR_CODE__', errcode_retriever)}\n".encode()
        )
        await self._process.stdin.drain()

        # read output from the process, until the sentinel is found
        try:
            async with asyncio.timeout(self._timeout):
                while True:
                    await asyncio.sleep(self._output_delay)
                    # if we read directly from stdout/stderr, it will wait forever for
                    # EOF. use the StreamReader buffer directly instead.
                    output: str = self._process.stdout._buffer.decode()  # type: ignore[attr-defined] # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
                    if sentinel_before in output:
                        # strip the sentinel from output
                        output, pivot, exit_banner = output.rpartition(sentinel_before)
                        assert pivot

                        # get error code inside banner
                        error_code_str, pivot, _ = exit_banner.partition(sentinel_after)
                        if not pivot or not error_code_str.isdecimal():
                            continue

                        error_code = int(error_code_str)
                        break
        except asyncio.TimeoutError:
            self._timed_out = True
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            ) from None

        if output.endswith("\n"):  # pyright: ignore[reportUnknownMemberType]
            output = output[:-1]  # pyright: ignore[reportUnknownVariableType]

        error: str = self._process.stderr._buffer.decode()  # type: ignore[attr-defined] # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]
        if error.endswith("\n"):  # pyright: ignore[reportUnknownMemberType]
            error = error[:-1]  # pyright: ignore[reportUnknownVariableType]

        # clear the buffers so that the next output can be read correctly
        self._process.stdout._buffer.clear()  # type: ignore[attr-defined] # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        self._process.stderr._buffer.clear()  # type: ignore[attr-defined] # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

        return ToolExecResult(output=output, error=error, error_code=error_code)  # pyright: ignore[reportUnknownArgumentType]


class _ContainerBashSession:
    """A session of a bash shell running in a container."""

    def __init__(self, executor: Executor):
        self.executor = executor
        self._session_id: str | None = None
        self._started: bool = False

    async def start(self) -> None:
        """Initialize a container session."""
        if self._started:
            return
        
        if not self.executor:
            raise ToolError("No executor provided for container operations")
        
        self._session_id = self.executor.init_session()
        if not self._session_id:
            raise ToolError("Failed to initialize container session")
        
        self._started = True

    async def stop(self) -> None:
        """Terminate the container session."""
        if not self._started or not self._session_id:
            return
        
        if self.executor:
            self.executor.close_session(self._session_id)
        
        self._session_id = None
        self._started = False

    async def run(self, command: str) -> ToolExecResult:
        """Execute a command in the container bash shell."""
        if not self._started or not self._session_id:
            raise ToolError("Container session has not started.")
        
        if not self.executor:
            raise ToolError("No executor available for container operations")

        try:
            return_code, output = self.executor.execute(self._session_id, command)
            
            # The executor returns (return_code, output) tuple
            # We'll treat any non-zero return code as an error
            error = None
            if return_code != 0:
                error = f"Command failed with exit code {return_code}"
            
            return ToolExecResult(
                output=output,
                error=error,
                error_code=return_code
            )
        except Exception as e:
            return ToolExecResult(
                error=f"Error executing command in container: {e}",
                error_code=-1
            )


class BashTool(Tool):
    """
    A tool that allows the agent to run bash commands.
    The tool parameters are defined by Anthropic and are not editable.
    """

    def __init__(self, model_provider: str | None = None, executor: Executor | None = None):
        super().__init__(model_provider)
        self._session: _BashSession | None = None
        self._container_session: _ContainerBashSession | None = None
        self.executor = executor

    @override
    def get_model_provider(self) -> str | None:
        return self._model_provider

    @override
    def get_name(self) -> str:
        return "bash"

    @override
    def get_description(self) -> str:
        return """Run commands in a bash shell (local or container)
* When invoking this tool, the contents of the "command" parameter does NOT need to be XML-escaped.
* You have access to a mirror of common linux and python packages via apt and pip.
* State is persistent across command calls and discussions with the user.
* To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.
* Please avoid commands that may produce a very large amount of output.
* Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.
* Use 'execute' for local execution or 'container_execute' for container execution (requires executor).
"""

    @override
    def get_parameters(self) -> list[ToolParameter]:
        # For OpenAI models, all parameters must be required=True
        # For other providers, optional parameters can have required=False
        restart_required = self.model_provider == "openai"

        return [
            ToolParameter(
                name="command",
                type="string",
                description="The bash command to run.",
                required=True,
            ),
            ToolParameter(
                name="restart",
                type="boolean",
                description="Set to true to restart the bash session.",
                required=restart_required,
            ),
        ]

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        if arguments.get("restart"):
            if self._session:
                await self._session.stop()
            self._session = _BashSession()
            await self._session.start()

            return ToolExecResult(output="tool has been restarted.")

        if self._session is None:
            try:
                self._session = _BashSession()
                await self._session.start()
            except Exception as e:
                return ToolExecResult(error=f"Error starting bash session: {e}", error_code=-1)

        command = str(arguments["command"]) if "command" in arguments else None
        if command is None:
            return ToolExecResult(
                error=f"No command provided for the {self.get_name()} tool",
                error_code=-1,
            )
        try:
            return await self._session.run(command)
        except Exception as e:
            return ToolExecResult(error=f"Error running bash command: {e}", error_code=-1)

    async def container_execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        """Execute a command in a container bash shell."""
        if not self.executor:
            return ToolExecResult(
                error="Container execution requires an executor to be provided during tool initialization",
                error_code=-1
            )

        if arguments.get("restart"):
            if self._container_session:
                await self._container_session.stop()
            self._container_session = _ContainerBashSession(self.executor)
            await self._container_session.start()
            return ToolExecResult(output="Container session has been restarted.")

        if self._container_session is None:
            try:
                self._container_session = _ContainerBashSession(self.executor)
                await self._container_session.start()
            except Exception as e:
                return ToolExecResult(error=f"Error starting container session: {e}", error_code=-1)

        command = str(arguments["command"]) if "command" in arguments else None
        if command is None:
            return ToolExecResult(
                error=f"No command provided for container execution",
                error_code=-1,
            )
        
        try:
            return await self._container_session.run(command)
        except Exception as e:
            return ToolExecResult(error=f"Error running container bash command: {e}", error_code=-1)

    @override
    async def close(self):
        """Properly close self._process and container session."""
        if self._session:
            await self._session.stop()
            self._session = None
        
        if self._container_session:
            await self._container_session.stop()
            self._container_session = None
