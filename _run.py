from typing import List, Dict, Any
from pathlib import Path
import yaml
import asyncio
from datetime import datetime

from src.tools.executor import Executor
from src.tools import BashTool, TextEditorTool, SearchTool
from src.tools.base import ToolExecutor
from src.managers.log.logger import create_logger, Logger
from src.managers.llm_api.api_manager import LLMAPIManager
from src.managers.image_builder.build_image import SWEBenchImageBuilder
from src.managers.prompts.prompts_manager import PromptsManager
from src.managers.loop.patch_generator import PatchGenerator

class SelectorLoop:
    def __init__(self, instance_id: str, image_name: str, runner_log_base: Path, llm_manager: LLMAPIManager | None, prompts_manager: PromptsManager | None, instance_data: Dict[str, Any], config: Dict[str, Any]):
        self.instance_id = instance_id
        self.image_name = image_name
        self.llm_manager = llm_manager
        self.prompts_manager = prompts_manager
        self.instance_data = instance_data
        self.config = config
        # SelectorLoop 日志目录：run/instance_id/selector/
        self.log_dir = runner_log_base / "run" / instance_id / "selector"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(
            log_base_path=str(self.log_dir.parent),
            logger_name=f"selector_{instance_id}",
            console_output=True,
            instance_id=self.log_dir.name
        )
    
    def select(self, generator_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """从 GeneratorLoop 结果中选择最佳结果"""
        import random
        
        self.logger.info(f"开始选择最佳结果，共 {len(generator_results)} 个候选")
        
        if not generator_results:
            self.logger.error("没有可选择的候选结果")
            return {
                "instance_id": self.instance_id,
                "generator_id": -1,
                "success": False,
                "error": "No candidates available",
                "llm_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "tool_stats": {"bash": 0, "edit": 0, "search": 0},
                "total_turns": 0,
            }
        
        # 过滤成功的候选结果
        successful_candidates = [r for r in generator_results if r.get("success", False)]
        
        if not successful_candidates:
            self.logger.warning("没有成功的候选结果，从所有候选中随机选择")
            candidates = generator_results
        else:
            self.logger.info(f"找到 {len(successful_candidates)} 个成功的候选结果")
            candidates = successful_candidates
        
        # 随机选择一个候选
        selected = random.choice(candidates)
        selected_generator_id = selected.get("generator_id", -1)
        
        self.logger.info(f"随机选择了 Generator #{selected_generator_id:03d}")
        
        # 构建选择结果
        result = {
            "instance_id": self.instance_id,
            "generator_id": selected_generator_id,
            "success": selected.get("success", False),
            "llm_usage": selected.get("llm_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
            "tool_stats": selected.get("tool_stats", {"bash": 0, "edit": 0, "search": 0}),
            "total_turns": selected.get("total_turns", 0),
        }
        
        if "error" in selected:
            result["error"] = selected["error"]
        
        self.logger.info(f"选择完成: Generator #{selected_generator_id:03d}, 成功: {result['success']}")
        return result

class GeneratorLoop:
    def __init__(self, instance_id: str, image_name: str, runner_log_base: Path, llm_manager: LLMAPIManager | None, prompts_manager: PromptsManager | None, instance_data: Dict[str, Any], config: Dict[str, Any], generator_id: int = 0):
        self.instance_id = instance_id
        self.image_name = image_name
        self.generator_id = generator_id
        self.llm_manager = llm_manager
        self.prompts_manager = prompts_manager
        self.instance_data = instance_data
        self.config = config
        # 每个实例的 generator 独立日志目录：run/instance_id/generator/generator_id/
        self.log_dir = runner_log_base / "run" / instance_id / "generator" / f"{generator_id:03d}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(
            log_base_path=str(self.log_dir.parent),
            logger_name=f"generator_{instance_id}_{generator_id:03d}",
            console_output=True,
            instance_id=self.log_dir.name
        )

    def generate(self) -> Dict[str, Any]:
        executor: Executor | None = None
        try:
            self.logger.info(f"启动实例 GeneratorLoop #{self.generator_id:03d}: {self.instance_id} -> {self.image_name}")
            executor = Executor(self.image_name)

            # 注册工具
            bash_tool = BashTool(model_provider=None, executor=executor)
            edit_tool = TextEditorTool(model_provider=None, executor=executor)
            search_tool = SearchTool(model_provider=None, executor=executor)
            tool_executor = ToolExecutor([bash_tool, edit_tool, search_tool])

            # 写Agent逻辑

            # 可选：做一次基本探活
            code, out = executor.execute('0', 'echo READY && rg --version || true')
            self.logger.info(f"容器探活: exit={code}, out=\n{out}")

            # 初始化 Patch 生成器，并执行一次候选补丁生成
            patch_generator = PatchGenerator(
                instance_id=self.instance_id,
                instance_data=self.instance_data,
                logger=self.logger,
                prompts_manager=self.prompts_manager,
                llm_manager=self.llm_manager,
                tool_executor=tool_executor,
                config=self.config,
            )
            _ = patch_generator._generate_patch()

            # Mock 结果结构体
            result: Dict[str, Any] = {
                "instance_id": self.instance_id,
                "generator_id": self.generator_id,
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
            self.logger.error(f"实例 {self.instance_id} Generator #{self.generator_id:03d} 失败: {e}")
            return {
                "instance_id": self.instance_id,
                "generator_id": self.generator_id,
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

        # 统一管理日志路径：基于时间戳创建主日志目录
        log_base_path = log_cfg.get("base_path", "workspace/logs")
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        self.logs_base = Path(log_base_path) / timestamp
        self.logs_base.mkdir(parents=True, exist_ok=True)
        
        # Runner 主日志器，直接放在时间戳文件夹下
        self.logger = Logger(
            log_base_path=str(self.logs_base.parent),
            logger_name="main",
            console_output=True,
            instance_id=self.logs_base.name
        )

        self.builder: SWEBenchImageBuilder | None = None
        # 初始化 LLM 管理器（取第一个provider）
        self.llm_manager = LLMAPIManager(logger=self.logger)
        
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

    # 运行单个 GeneratorLoop
    async def _run_one(self, instance_id: str, image_name: str, instance_data: Dict[str, Any], generator_id: int = 0) -> Dict[str, Any]:
        loop = GeneratorLoop(instance_id, image_name, self.logs_base, self.llm_manager, self.prompts_manager, instance_data, self.cfg, generator_id)
        # 在线程池中执行阻塞型工作，便于并发
        return await asyncio.to_thread(loop.generate)

    # 运行单个 instance
    async def process_one_instance(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个实例"""
        instance_id = instance["instance_id"]
        try:
            image_name = self.builder.get_image_name(instance_id)
        except KeyError:
            self.logger.warning(f"跳过实例（未找到镜像映射）: {instance_id}")
            return None

        self.logger.info(f"开始处理实例: {instance_id}")

        # 1.并发生成所有候选 patch
        generator_tasks = []
        for generator_id in range(self.generator_loop_concurrency):
            task = asyncio.create_task(self._run_one(instance_id, image_name, instance, generator_id))
            generator_tasks.append(task)

        # 2.等待该实例的所有 GeneratorLoop 完成
        generator_results = await asyncio.gather(*generator_tasks, return_exceptions=True)
        
        # 3.过滤异常结果
        valid_results = []
        for result in generator_results:
            if isinstance(result, Exception):
                self.logger.error(f"GeneratorLoop 异常: {result}")
            else:
                valid_results.append(result)

        if not valid_results:
            self.logger.warning(f"实例 {instance_id} 没有有效的 GeneratorLoop 结果")
            return None

        # 4.patch选取
        self.logger.info(f"开始为实例 {instance_id} 选择最佳结果")
        selector = SelectorLoop(
            instance_id=instance_id,
            image_name=image_name,
            runner_log_base=self.logs_base,
            llm_manager=self.llm_manager,
            prompts_manager=self.prompts_manager,
            instance_data=instance,
            config=self.cfg,
        )
        selected_result = selector.select(valid_results) # 选择最佳结果


        self.logger.info(f"实例 {instance_id} 处理完成，选择了 Generator #{selected_result.get('generator_id', -1):03d}")
        return selected_result

    async def run(self) -> Dict[str, Any]:
        # 入口函数-串行执行所有实例

        # 1.构建镜像
        if self.builder is None:
            self.build_images()

        assert self.builder is not None

        # 仅运行指定的实例（如果提供了 instance_ids），否则跑全量
        if self.instance_ids:
            target_ids = set(self.instance_ids)
            instances_to_run = [inst for inst in self.builder.full_dataset if inst.get("instance_id") in target_ids]
        else:
            instances_to_run = list(self.builder.full_dataset)

        self.logger.info(f"开始运行 {len(instances_to_run)} 个实例")

        # 2. 串行处理目标实例
        final_results = []
        for i, instance in enumerate(instances_to_run, 1):
            self.logger.info(f"处理实例 {i}/{len(instances_to_run)}: {instance['instance_id']}")
            try:
                # 处理单个实例
                result = await self.process_one_instance(instance)
                if result is not None:
                    final_results.append(result)
            except Exception as e:
                self.logger.error(f"实例 {instance['instance_id']} 处理异常: {e}")
                # 可以选择添加一个失败结果到 final_results 中
        
        # 3. 结果统计
        summary = self._calculate_summary(final_results)

        self.logger.info(f"完成。总数={summary['total']} 成功={summary['success']} 失败={summary['failed']}")
        return summary

    def _calculate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算运行结果统计摘要"""
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
        
        return summary


def main() -> None:
    # 加载配置
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # 测试用例，置空则跑全量
    test_instance_ids = ["sqlfluff__sqlfluff-4764"]

    runner = Runner(cfg, test_instance_ids)
    summary = asyncio.run(runner.run())
    
    # 打印结果
    print("\n" + "="*80)
    print("运行结果汇总")
    print("="*80)
    print(f"总实例数: {summary['total']}")
    print(f"成功: {summary['success']}")
    print(f"失败: {summary['failed']}")
    print(f"成功率: {(summary['success']/summary['total']*100):.1f}%" if summary['total'] > 0 else "0%")
    
    print("\n详细结果:")
    print("-" * 80)
    for result in summary['details']:
        instance_id = result.get('instance_id', 'unknown')
        generator_id = result.get('generator_id', -1)
        success = result.get('success', False)
        llm_usage = result.get('llm_usage', {})
        tool_stats = result.get('tool_stats', {})
        total_turns = result.get('total_turns', 0)
        
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{instance_id} -> Generator #{generator_id:03d} {status}")
        print(f"  LLM使用: {llm_usage.get('total_tokens', 0)} tokens")
        print(f"  工具统计: bash={tool_stats.get('bash', 0)}, edit={tool_stats.get('edit', 0)}, search={tool_stats.get('search', 0)}")
        print(f"  总轮数: {total_turns}")
        if not success and 'error' in result:
            print(f"  错误: {result['error']}")
        print()
    
    print("="*80)


if __name__ == "__main__":
    main()


