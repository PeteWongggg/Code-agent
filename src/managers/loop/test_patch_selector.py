#!/usr/bin/env python3
"""
Test script for Patch Selector with LLM tool calls
Follows the pattern from _run.py
"""

import asyncio
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from src.managers.llm_api.api_manager import LLMAPIManager
from src.managers.prompts.prompts_manager import PromptsManager
from src.managers.patch_selector.test_loop import run_patch_selector_test
from src.managers.log.logger import Logger


class PatchSelectorTestRunner:
    """Test runner for patch selector functionality."""
    
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        
        # Setup logging
        log_cfg = cfg.get("log", {})
        log_base_path = log_cfg.get("base_path", "workspace/logs")
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        self.logs_base = Path(log_base_path) / f"patch_selector_test_{timestamp}"
        self.logs_base.mkdir(parents=True, exist_ok=True)
        
        self.logger = Logger(
            log_base_path=str(self.logs_base.parent),
            logger_name="patch_selector_test",
            console_output=True,
            instance_id=self.logs_base.name
        )
        
        # Initialize LLM manager
        providers_cfg = cfg.get("providers", {})
        self.llm_manager = None
        try:
            first_provider = next(iter(providers_cfg.keys())) if providers_cfg else None
            if first_provider:
                self.llm_manager = LLMAPIManager(client_name=first_provider, logger=self.logger)
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM manager: {e}")
        
        # Initialize Prompts manager
        self.prompts_manager = None
        try:
            self.prompts_manager = PromptsManager(cfg)
        except Exception as e:
            self.logger.error(f"Failed to initialize PromptsManager: {e}")
    
    async def run_test(self, instance_id: str = "test-instance", image_name: str = "test-image") -> Dict[str, Any]:
        """Run a single patch selector test."""
        if not self.llm_manager:
            return {
                "success": False,
                "error": "LLM manager not initialized"
            }
        
        if not self.prompts_manager:
            return {
                "success": False,
                "error": "Prompts manager not initialized"
            }
        
        # Mock instance data
        instance_data = {
            "instance_id": instance_id,
            "problem_statement": "Test problem: function should multiply by 3 instead of 2",
            "repo": "test-repo",
            "base_commit": "abc123"
        }
        
        self.logger.info(f"Starting patch selector test for instance: {instance_id}")
        
        try:
            result = await run_patch_selector_test(
                instance_id=instance_id,
                image_name=image_name,
                runner_log_base=self.logs_base,
                llm_manager=self.llm_manager,
                prompts_manager=self.prompts_manager,
                instance_data=instance_data
            )
            
            self.logger.info(f"Test completed: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Test failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def run_multiple_tests(self, num_tests: int = 3) -> List[Dict[str, Any]]:
        """Run multiple patch selector tests."""
        results = []
        
        for i in range(num_tests):
            instance_id = f"test-instance-{i+1}"
            image_name = f"test-image-{i+1}"
            
            self.logger.info(f"Running test {i+1}/{num_tests}")
            result = await self.run_test(instance_id, image_name)
            results.append(result)
            
            # Small delay between tests
            await asyncio.sleep(1)
        
        return results


async def main():
    """Main test function."""
    # Load configuration
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    
    # Create test runner
    runner = PatchSelectorTestRunner(cfg)
    
    # Run single test
    print("Running single patch selector test...")
    result = await runner.run_test()
    
    print("\n" + "="*80)
    print("Single Test Results")
    print("="*80)
    print(f"Success: {result.get('success', False)}")
    if result.get('success'):
        print(f"Selected Patch ID: {result.get('selected_patch_id')}")
        print(f"LLM Usage: {result.get('llm_usage', {})}")
        print(f"Tool Stats: {result.get('tool_stats', {})}")
        print(f"Total Turns: {result.get('total_turns', 0)}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Run multiple tests
    print("\nRunning multiple tests...")
    results = await runner.run_multiple_tests(num_tests=2)
    
    print("\n" + "="*80)
    print("Multiple Test Results")
    print("="*80)
    
    successful_tests = 0
    total_llm_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    total_tool_stats = {"bash": 0, "edit": 0, "search": 0}
    
    for i, result in enumerate(results, 1):
        print(f"\nTest {i}:")
        print(f"  Success: {result.get('success', False)}")
        if result.get('success'):
            successful_tests += 1
            print(f"  Selected Patch ID: {result.get('selected_patch_id')}")
            
            # Aggregate statistics
            llm_usage = result.get('llm_usage', {})
            total_llm_usage['prompt_tokens'] += llm_usage.get('prompt_tokens', 0)
            total_llm_usage['completion_tokens'] += llm_usage.get('completion_tokens', 0)
            total_llm_usage['total_tokens'] += llm_usage.get('total_tokens', 0)
            
            tool_stats = result.get('tool_stats', {})
            total_tool_stats['bash'] += tool_stats.get('bash', 0)
            total_tool_stats['edit'] += tool_stats.get('edit', 0)
            total_tool_stats['search'] += tool_stats.get('search', 0)
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")
    
    print(f"\nSummary:")
    print(f"  Total Tests: {len(results)}")
    print(f"  Successful: {successful_tests}")
    print(f"  Success Rate: {(successful_tests/len(results)*100):.1f}%")
    print(f"  Total LLM Usage: {total_llm_usage}")
    print(f"  Total Tool Stats: {total_tool_stats}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(main())
