"""
Bash 命令执行工具
提供执行 bash 命令并获取结果的功能，支持 LLM 生成和执行命令
"""

import subprocess
import os
import time
import threading
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum


class ExecutionStatus(Enum):
    """命令执行状态枚举"""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class BashResult:
    """Bash 命令执行结果"""
    command: str                    # 执行的命令
    status: ExecutionStatus         # 执行状态
    return_code: int               # 返回码
    stdout: str                    # 标准输出
    stderr: str                    # 标准错误输出
    execution_time: float          # 执行时间（秒）
    working_directory: str         # 工作目录
    environment_vars: Dict[str, str] = None  # 环境变量快照
    
    @property
    def success(self) -> bool:
        """是否执行成功"""
        return self.status == ExecutionStatus.SUCCESS and self.return_code == 0
    
    @property
    def output(self) -> str:
        """获取完整输出（stdout + stderr）"""
        output_parts = []
        if self.stdout:
            output_parts.append(f"STDOUT:\n{self.stdout}")
        if self.stderr:
            output_parts.append(f"STDERR:\n{self.stderr}")
        return "\n\n".join(output_parts) if output_parts else ""
    
    def __str__(self) -> str:
        """字符串表示"""
        return (
            f"BashResult(\n"
            f"  command='{self.command}'\n"
            f"  status={self.status.value}\n"
            f"  return_code={self.return_code}\n"
            f"  execution_time={self.execution_time:.3f}s\n"
            f"  stdout_length={len(self.stdout)}\n"
            f"  stderr_length={len(self.stderr)}\n"
            f")"
        )


class BashTool:
    """
    Bash 命令执行工具类
    
    提供安全的 bash 命令执行功能，支持：
    - 命令执行和结果获取
    - 超时控制
    - 工作目录设置
    - 环境变量管理
    - 安全检查
    - 执行历史记录
    
    兼容性说明：
    - 在 Unix 系统（posix）上，使用系统默认 shell（/bin/sh）执行；
    - 在 Windows 上，subprocess 配置为 shell=True 时会调用 cmd.exe 处理命令；
      因此本工具在两类系统上均可直接工作，无需额外适配参数。
    """
    
    # 危险命令黑名单（可配置）- 移除了 rm -rf 相关的通用匹配
    DANGEROUS_COMMANDS = {
        'rm -rf /', 'rm -rf /*', 'rm -rf ~', 'rm -rf *',
        'dd if=/dev/zero', 'dd if=/dev/random',
        'mkfs', 'fdisk', 'parted',
        'shutdown', 'reboot', 'halt', 'poweroff',
        'kill -9 -1', 'killall -9',
        'chmod 777 /', 'chown root:root /',
        ':(){ :|:& };:', # fork bomb
        'curl | bash', 'wget | bash', 'curl | sh', 'wget | sh'
    }
    
    def __init__(
        self,
        working_directory: Optional[str] = None,
        timeout: int = 30,
        max_output_size: int = 1024 * 1024,  # 1MB
        enable_safety_check: bool = True,
        skip_safety_check: Optional[bool] = None,
        allowed_commands: Optional[List[str]] = None,
        blocked_commands: Optional[List[str]] = None,
        logger: Optional[Any] = None
    ):
        """
        初始化 Bash 工具
        
        Args:
            working_directory: 工作目录，默认为当前目录
            timeout: 命令执行超时时间（秒）
            max_output_size: 最大输出大小（字节）
            enable_safety_check: 是否启用安全检查
            allowed_commands: 允许的命令列表（白名单）
            blocked_commands: 禁止的命令列表（黑名单）
            logger: 日志记录器
        """
        self.working_directory = working_directory or os.getcwd()
        self.timeout = timeout
        self.max_output_size = max_output_size
        # 支持通过 skip_safety_check 快速关闭安全检查（优先级高于 enable_safety_check）
        if skip_safety_check is not None:
            self.enable_safety_check = not bool(skip_safety_check)
        else:
            self.enable_safety_check = enable_safety_check
        self.allowed_commands = set(allowed_commands or [])
        self.blocked_commands = set(blocked_commands or [])
        self.logger = logger
        
        # 执行历史
        self.execution_history: List[BashResult] = []
        self.max_history_size = 100
        
        # 确保工作目录存在
        if not os.path.exists(self.working_directory):
            try:
                os.makedirs(self.working_directory, exist_ok=True)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"无法创建工作目录 {self.working_directory}: {e}")
    
    def _log(self, level: str, message: str) -> None:
        """记录日志"""
        if self.logger:
            getattr(self.logger, level)(f"[BashTool] {message}")
    
    def _is_command_safe(self, command: str) -> tuple[bool, str]:
        """
        检查命令是否安全
        
        Args:
            command: 要检查的命令
            
        Returns:
            tuple[bool, str]: (是否安全, 原因)
        """
        if not self.enable_safety_check:
            return True, "安全检查已禁用"
        
        command_lower = command.lower().strip()
        
        # 检查黑名单（精确匹配，避免误判）
        for dangerous in self.DANGEROUS_COMMANDS:
            dangerous_lower = dangerous.lower()
            # 对于 rm -rf 命令，需要精确匹配以避免误判安全路径
            if dangerous_lower.startswith('rm -rf'):
                if dangerous_lower == command_lower or f"{dangerous_lower} " in f"{command_lower} ":
                    return False, f"命令包含危险操作: {dangerous}"
            else:
                # 其他命令使用包含匹配
                if dangerous_lower in command_lower:
                    return False, f"命令包含危险操作: {dangerous}"
        
        # 检查自定义黑名单
        for blocked in self.blocked_commands:
            if blocked.lower() in command_lower:
                return False, f"命令被明确禁止: {blocked}"
        
        # 如果设置了白名单，检查是否在白名单中
        if self.allowed_commands:
            command_parts = command_lower.split()
            if command_parts:
                base_command = command_parts[0]
                if base_command not in [cmd.lower() for cmd in self.allowed_commands]:
                    return False, f"命令不在允许列表中: {base_command}"
        
        # 检查一些危险模式（更精确的检查）
        dangerous_patterns = [
            ('sudo rm', '使用 sudo 删除文件'),
            ('> /dev/', '重定向到设备文件'),
            ('dd if=', 'dd 命令可能危险'),
            ('mkfs.', '格式化文件系统'),
            ('fdisk', '磁盘分区操作'),
            ('chmod 777', '设置过于宽松的权限'),
            ('chmod -R 777', '递归设置过于宽松的权限'),
            ('chown -R root', '递归更改所有权为 root'),
            ('| bash', '管道到 bash 执行'),
            ('| sh', '管道到 sh 执行'),
        ]
        
        for pattern, reason in dangerous_patterns:
            if pattern in command_lower:
                return False, f"命令包含危险模式: {reason}"
        
        # 特殊检查 rm -rf，但允许安全路径
        if 'rm -rf' in command_lower:
            import re
            # 提取 rm -rf 的目标路径
            rm_target_match = re.search(r'rm\s+-rf\s+([^\s]+)', command_lower)
            if rm_target_match:
                target = rm_target_match.group(1)
                
                # 首先检查是否是明确的危险路径
                dangerous_exact_patterns = [
                    '/', '/*', '~', '*', '$home', '/usr', '/etc', '/var', '/opt', 
                    '/boot', '/sys', '/proc', '/bin', '/sbin', '/lib', '/root'
                ]
                
                if target in dangerous_exact_patterns:
                    return False, f"危险的删除操作: rm -rf {target}"
                
                # 检查是否是安全路径
                safe_rm_patterns = [
                    '/tmp/', '/var/tmp/', './temp', './build', './dist', 
                    './node_modules', '__pycache__', 'temp', 'build', 'dist'
                ]
                
                # 如果目标路径包含安全模式，允许执行
                is_safe_rm = any(safe_pattern in target for safe_pattern in safe_rm_patterns)
                
                # 或者如果是相对路径（不以 / 开头），也允许
                is_relative_path = not target.startswith('/')
                
                if is_safe_rm or is_relative_path:
                    # 这是安全的删除操作，继续其他检查
                    pass
                else:
                    # 绝对路径且不在安全列表中，需要更仔细的检查
                    # 检查是否是系统重要目录的子路径
                    dangerous_prefixes = ['/usr/', '/etc/', '/var/', '/opt/', '/boot/', '/sys/', '/proc/']
                    if any(target.startswith(prefix) for prefix in dangerous_prefixes):
                        return False, f"不安全的系统目录删除: {target}"
        
        return True, "命令通过安全检查"
    
    def execute(
        self,
        command: str,
        timeout: Optional[int] = None,
        working_directory: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        capture_output: bool = True,
        shell: bool = True
    ) -> BashResult:
        """
        执行 bash 命令
        
        Args:
            command: 要执行的命令
            timeout: 超时时间（秒），覆盖默认值
            working_directory: 工作目录，覆盖默认值
            environment: 环境变量，会合并到当前环境
            capture_output: 是否捕获输出
            shell: 是否通过 shell 执行
            
        Returns:
            BashResult: 执行结果
        """
        start_time = time.time()
        actual_timeout = timeout or self.timeout
        actual_working_dir = working_directory or self.working_directory
        
        self._log("info", f"开始执行命令: {command}")
        self._log("debug", f"工作目录: {actual_working_dir}, 超时: {actual_timeout}s")
        
        # 安全检查
        is_safe, safety_reason = self._is_command_safe(command)
        if not is_safe:
            self._log("error", f"命令被安全检查拒绝: {safety_reason}")
            result = BashResult(
                command=command,
                status=ExecutionStatus.ERROR,
                return_code=-1,
                stdout="",
                stderr=f"安全检查失败: {safety_reason}",
                execution_time=0.0,
                working_directory=actual_working_dir,
                environment_vars=environment
            )
            self._add_to_history(result)
            return result
        
        # 准备环境变量
        env = os.environ.copy()
        if environment:
            env.update(environment)
        
        try:
            # 执行命令
            process = subprocess.Popen(
                command,
                shell=shell,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None,
                cwd=actual_working_dir,
                env=env,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=actual_timeout)
                return_code = process.returncode
                status = ExecutionStatus.SUCCESS if return_code == 0 else ExecutionStatus.ERROR
                
                # 限制输出大小
                if stdout and len(stdout) > self.max_output_size:
                    stdout = stdout[:self.max_output_size] + f"\n... (输出被截断，原始长度: {len(stdout)} 字节)"
                
                if stderr and len(stderr) > self.max_output_size:
                    stderr = stderr[:self.max_output_size] + f"\n... (错误输出被截断，原始长度: {len(stderr)} 字节)"
                
            except subprocess.TimeoutExpired:
                self._log("warning", f"命令执行超时 ({actual_timeout}s)")
                process.kill()
                stdout, stderr = process.communicate()
                return_code = -1
                status = ExecutionStatus.TIMEOUT
                stderr = f"命令执行超时 ({actual_timeout}s)\n" + (stderr or "")
                
        except Exception as e:
            self._log("error", f"命令执行异常: {str(e)}")
            stdout = ""
            stderr = f"执行异常: {str(e)}"
            return_code = -1
            status = ExecutionStatus.ERROR
        
        execution_time = time.time() - start_time
        
        # 创建结果对象
        result = BashResult(
            command=command,
            status=status,
            return_code=return_code,
            stdout=stdout or "",
            stderr=stderr or "",
            execution_time=execution_time,
            working_directory=actual_working_dir,
            environment_vars=environment
        )
        
        self._log("info", f"命令执行完成: {status.value}, 返回码: {return_code}, 耗时: {execution_time:.3f}s")
        
        # 添加到历史记录
        self._add_to_history(result)
        
        return result
    
    def _add_to_history(self, result: BashResult) -> None:
        """添加执行结果到历史记录"""
        self.execution_history.append(result)
        
        # 限制历史记录大小
        if len(self.execution_history) > self.max_history_size:
            self.execution_history = self.execution_history[-self.max_history_size:]
    
    def get_history(self, limit: Optional[int] = None) -> List[BashResult]:
        """
        获取执行历史
        
        Args:
            limit: 返回的记录数量限制
            
        Returns:
            List[BashResult]: 历史记录列表
        """
        if limit:
            return self.execution_history[-limit:]
        return self.execution_history.copy()
    
    def clear_history(self) -> None:
        """清空执行历史"""
        self.execution_history.clear()
        self._log("info", "执行历史已清空")
    
    def execute_multiple(
        self,
        commands: List[str],
        stop_on_error: bool = True,
        **kwargs
    ) -> List[BashResult]:
        """
        批量执行多个命令
        
        Args:
            commands: 命令列表
            stop_on_error: 是否在出错时停止
            **kwargs: 传递给 execute 方法的参数
            
        Returns:
            List[BashResult]: 执行结果列表
        """
        results = []
        
        for i, command in enumerate(commands):
            self._log("info", f"执行批量命令 {i+1}/{len(commands)}: {command}")
            
            result = self.execute(command, **kwargs)
            results.append(result)
            
            if stop_on_error and not result.success:
                self._log("warning", f"命令执行失败，停止批量执行: {command}")
                break
        
        return results
    
    def test_command(self, command: str) -> Dict[str, Any]:
        """
        测试命令（仅进行安全检查，不实际执行）
        
        Args:
            command: 要测试的命令
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        is_safe, reason = self._is_command_safe(command)
        
        return {
            "command": command,
            "is_safe": is_safe,
            "safety_reason": reason,
            "would_execute": is_safe
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取执行统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not self.execution_history:
            return {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "timeout_executions": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0
            }
        
        successful = sum(1 for r in self.execution_history if r.success)
        failed = sum(1 for r in self.execution_history if r.status == ExecutionStatus.ERROR)
        timeout = sum(1 for r in self.execution_history if r.status == ExecutionStatus.TIMEOUT)
        total = len(self.execution_history)
        avg_time = sum(r.execution_time for r in self.execution_history) / total
        
        return {
            "total_executions": total,
            "successful_executions": successful,
            "failed_executions": failed,
            "timeout_executions": timeout,
            "success_rate": (successful / total) * 100,
            "average_execution_time": avg_time
        }
    
    def get_name(self) -> str:
        """工具名称（用于工具注册）"""
        return "bash"

    def get_description(self) -> Dict[str, Any]:
        """
        返回可直接用于大模型调用的工具描述（OpenAI functions 兼容结构）。
        可放入 chat.completions/tools 数组中使用。
        
        Returns:
            Dict[str, Any]: 工具描述结构体
        """
        # 说明中标注跨平台行为，指导模型合理使用
        long_desc = (
            "Run commands in a bash shell (cross-platform). "
            "On Unix, commands execute under the system shell; on Windows, under cmd.exe. "
            "Avoid extremely verbose output. Prefer short, atomic commands. "
            "For long-running tasks, consider backgrounding with '&'."
        )
        return {
            "type": "function",
            "function": {
                "name": self.get_name(),
                "description": long_desc,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": (
                                "The bash command to run. Example: 'ls -la' or "
                                "'python -c \"print(123)\"'."
                            ),
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Optional timeout in seconds for this command.",
                        },
                        "working_directory": {
                            "type": "string",
                            "description": "Optional working directory to run the command in.",
                        },
                        "capture_output": {
                            "type": "boolean",
                            "description": "Capture stdout/stderr (default: true).",
                        },
                    },
                    "required": ["command"],
                },
            },
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        stats = self.get_stats()
        return (
            f"BashTool(\n"
            f"  working_directory='{self.working_directory}'\n"
            f"  timeout={self.timeout}s\n"
            f"  safety_check={'enabled' if self.enable_safety_check else 'disabled'}\n"
            f"  executions={stats['total_executions']}\n"
            f"  success_rate={stats['success_rate']:.1f}%\n"
            f")"
        )


if __name__ == "__main__":
    # 简单测试：跳过安全检查，在 logs 目录中删除所有文件与子目录
    import os
    import sys

    # 计算 Code-agent 根目录与 logs 目录
    current_file = os.path.abspath(__file__)
    # src/tools/bash_tool.py → Code-agent
    code_agent_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    logs_dir = os.path.join(code_agent_root, "logs")

    print("[BashTool __main__] 目标 logs 目录:", logs_dir)
    if not os.path.isdir(logs_dir):
        print("[BashTool __main__] 未找到 logs 目录，退出")
        sys.exit(1)

    # 跳过安全检查实例
    tool = BashTool(
        working_directory=logs_dir,
        timeout=30,
        skip_safety_check=True,
    )

    # 删除前查看
    pre = tool.execute("pwd && echo 'before:' && ls -la || true")
    print("--- BEFORE ---")
    print(pre.output)

    # 使用 find 删除 logs 内的所有条目（包含隐藏文件与子目录）
    # -mindepth 1 确保不会删除当前目录本身
    cmd = "find . -mindepth 1 -exec rm -rf {} +"
    res = tool.execute(cmd)
    print("--- DELETE ---")
    print(res.output if res.output else res.stderr)

    # 删除后查看
    post = tool.execute("echo 'after:' && ls -la || true")
    print("--- AFTER ---")
    print(post.output)

    print("[BashTool __main__] 清理完成")
