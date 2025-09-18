from typing import List, Dict, Any
from pathlib import Path
import yaml
import asyncio

from src.tools.executor import Executor
from src.tools import BashTool, TextEditorTool, SearchTool
from src.tools.base import ToolExecutor
from src.managers.log.logger import create_logger
from src.managers.llm_api.api_manager import LLMAPIManager
from src.managers.image_builder.build_image import SWEBenchImageBuilder
from src.managers.prompts.prompts_manager import PromptsManager

class GeneratorLoop:
    def __init__(self, instance_id: str, image_name: str, runner_log_base: Path, llm_manager: LLMAPIManager | None, prompts_manager: PromptsManager | None, instance_data: Dict[str, Any]):
        self.instance_id = instance_id
        self.image_name = image_name
        self.llm_manager = llm_manager
        self.prompts_manager = prompts_manager
        self.instance_data = instance_data
        # 每个实例独立日志目录（位于 Runner 的 log 路径下）
        self.log_dir = runner_log_base / instance_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = create_logger(log_base_path=str(self.log_dir), logger_name=f"loop_{instance_id}", console_output=True)

    def run(self) -> Dict[str, Any]:
        executor: Executor | None = None
        try:
            self.logger.info(f"启动实例 GeneratorLoop: {self.instance_id} -> {self.image_name}")
            executor = Executor(self.image_name)

            # 注册工具
            bash_tool = BashTool(model_provider=None, executor=executor)
            edit_tool = TextEditorTool(model_provider=None, executor=executor)
            search_tool = SearchTool(model_provider=None, executor=executor)
            _ = ToolExecutor([bash_tool, edit_tool, search_tool])

            # 写Agent逻辑

            # 可选：做一次基本探活
            code, out = executor.execute('0', 'echo READY && rg --version || true')
            self.logger.info(f"容器探活: exit={code}, out=\n{out}")

            # Mock 结果结构体
            result: Dict[str, Any] = {
                "instance_id": self.instance_id,
                "image": self.image_name,
                "success": True,
                "golden_patch": [],
                "llm_usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "tool_stats": {
                    "bash": 1,   # 探活一次
                    "edit": 0,
                    "search": 0,
                },
                "total_turns": 0,
            }

            return result
        except Exception as e:
            self.logger.error(f"实例 {self.instance_id} 失败: {e}")
            return {
                "instance_id": self.instance_id,
                "image": self.image_name,
                "success": False,
                "error": str(e),
                "golden_patch": [],
                "llm_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "tool_stats": {"bash": 0, "edit": 0, "search": 0},
                "total_turns": 0,
            }
        finally:
            if executor:
                try:
                    executor.shutdown()
                except Exception:
                    pass


class Runner:
    def __init__(self, cfg: Dict[str, Any], instance_ids: List[str] = None):
        self.cfg = cfg
        dataset_cfg = cfg.get("dataset", {})
        workspace_cfg = cfg.get("workspace", {})
        builder_cfg = cfg.get("builder", {})
        log_cfg = cfg.get("log", {})
        runner_cfg = cfg.get("runner", {})
        providers_cfg = cfg.get("providers", {})
        self.instance_ids = instance_ids
        self.dataset_name = dataset_cfg.get("name", "princeton-nlp/SWE-bench_Lite")
        self.dataset_split = dataset_cfg.get("split", "dev")
        self.max_workers = int(builder_cfg.get("max_workers", 2)) # 镜像构建并发数量
        self.generator_loop_concurrency = int(runner_cfg.get("generator_concurrency", 2)) # GeneratorLoop 并发数量

        # 日志路径优先级：log.path > workspace.log_path > workspace.path/logs/run
        self.logs_base = Path(
            log_cfg.get(
                "path",
                workspace_cfg.get(
                    "log_path",
                    str(Path(workspace_cfg.get("path", "workspace")) / "logs" / "run")
                ),
            )
        )
        self.logs_base.mkdir(parents=True, exist_ok=True)
        self.logger = create_logger(log_base_path=str(self.logs_base), logger_name="swebench_runner", console_output=True)

        self.builder: SWEBenchImageBuilder | None = None
        # 初始化 LLM 管理器（取第一个可用 provider）
        self.llm_manager: LLMAPIManager | None = None
        try:
            first_provider: str | None = next(iter(providers_cfg.keys())) if isinstance(providers_cfg, dict) and providers_cfg else None
            if first_provider:
                self.llm_manager = LLMAPIManager(client_name=first_provider, logger=self.logger)
        except Exception:
            self.llm_manager = None
        
        # 初始化 Prompts 管理器
        self.prompts_manager: PromptsManager | None = None
        try:
            self.prompts_manager = PromptsManager(cfg)
        except Exception as e:
            self.logger.warning(f"Failed to initialize PromptsManager: {e}")
            self.prompts_manager = None

    def build_images(self):
        self.logger.info("初始化 SWEBenchImageBuilder 并准备镜像...")
        self.builder = SWEBenchImageBuilder(
            dataset_name=self.dataset_name,
            split=self.dataset_split,
            instance_ids=self.instance_ids,
            max_workers=self.max_workers,
            force_rebuild=False,
            namespace=None,
            tag="latest",
            env_image_tag="latest",
        )

    async def _run_one(self, instance_id: str, image_name: str, instance_data: Dict[str, Any]) -> Dict[str, Any]:
        loop = GeneratorLoop(instance_id, image_name, self.logs_base, self.llm_manager, self.prompts_manager, instance_data)
        # 在线程池中执行阻塞型工作，便于并发
        return await asyncio.to_thread(loop.run)

    async def run(self) -> Dict[str, Any]:
        if self.builder is None:
            self.build_images()

        assert self.builder is not None

        sem = asyncio.Semaphore(self.generator_loop_concurrency)
        tasks = []

        async def bound_run(instance_id: str, image_name: str, instance_data: Dict[str, Any]):
            async with sem:
                return await self._run_one(instance_id, image_name, instance_data)

        self.logger.info(f"开始运行 {len(self.builder.full_dataset)} 个实例")

        for instance in self.builder.full_dataset:
            instance_id = instance["instance_id"]
            try:
                image_name = self.builder.get_image_name(instance_id)
            except KeyError:
                self.logger.warning(f"跳过实例（未找到镜像映射）: {instance_id}")
                continue
            tasks.append(asyncio.create_task(bound_run(instance_id, image_name, instance)))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        summary: Dict[str, Any] = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "details": [],
            "aggregate": {
                "llm_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "tool_stats": {"bash": 0, "edit": 0, "search": 0},
                "total_turns": 0,
            },
        }
        for r in results:
            if isinstance(r, Exception):
                self.logger.error(f"并发任务异常: {r}")
                summary["failed"] += 1
                summary["total"] += 1
                continue
            summary["total"] += 1
            if r.get("success"):
                summary["success"] += 1
            else:
                summary["failed"] += 1
            # 汇总统计
            usage = r.get("llm_usage", {})
            stats = r.get("tool_stats", {})
            summary["aggregate"]["llm_usage"]["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
            summary["aggregate"]["llm_usage"]["completion_tokens"] += int(usage.get("completion_tokens", 0))
            summary["aggregate"]["llm_usage"]["total_tokens"] += int(usage.get("total_tokens", 0))
            summary["aggregate"]["tool_stats"]["bash"] += int(stats.get("bash", 0))
            summary["aggregate"]["tool_stats"]["edit"] += int(stats.get("edit", 0))
            summary["aggregate"]["tool_stats"]["search"] += int(stats.get("search", 0))
            summary["aggregate"]["total_turns"] += int(r.get("total_turns", 0))
            summary["details"].append(r)

        self.logger.info(f"完成。总数={summary['total']} 成功={summary['success']} 失败={summary['failed']}")
        return summary


def main() -> None:
    # 加载配置
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    test_instance_ids = ["sqlfluff__sqlfluff-4764"]

    runner = Runner(cfg, test_instance_ids)
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()


