"""
Patch Selector for Code-agent
Adapted from Trae-agent's patch selection logic

This module provides intelligent patch selection capabilities that can:
- Analyze multiple candidate patches
- Evaluate patches against issue descriptions
- Use LLM-based reasoning for selection
- Support majority voting and consensus mechanisms
"""

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import tempfile
import os
import re

from ..llm_api.api_manager import LLMAPIManager
from ...tools.base import Tool, ToolCall, ToolResult, ToolExecutor
from ...tools.executor import Executor
from .base import BaseLoopManager


@dataclass
class CandidatePatch:
    """Represents a candidate patch with metadata.
    
    Assumes the patch is proper git diff output that can be applied directly.
    """
    
    id: int
    patch: str  # Proper git diff output
    is_success_regression: bool = True
    is_success_patch: bool = False
    source: str = "unknown"  # Where this patch came from
    confidence: float = 0.0  # Confidence score for this patch


class PatchSelector(BaseLoopManager):
    """
    Intelligent patch selector that uses LLM reasoning to choose the best patch
    from multiple candidates.
    """
    
    def __init__(
        self,
        llm_manager: LLMAPIManager,
        model: str,
        image_name: str,
        instance_id: str,
        max_turns: int = 50,
        temperature: float = 0.7,
        tools: Optional[List[Tool]] = None,
        logger: Optional[Any] = None
    ):
        """
        Initialize the patch selector.
        
        Args:
            llm_manager: LLM API manager for making requests
            model: Model name to use for selection
            image_name: Docker image name for container-based testing (required)
            instance_id: Instance ID for tracking (required)
            max_turns: Maximum number of reasoning turns
            temperature: LLM temperature for generation
            tools: Optional tools for the selector to use
            logger: Logger instance
        """
        super().__init__(
            llm_manager=llm_manager,
            model=model,
            image_name=image_name,
            instance_id=instance_id,
            max_turns=max_turns,
            temperature=temperature,
            tools=tools,
            logger=logger
        )
    
    def build_system_prompt(self, candidate_count: int = None, **kwargs) -> str:
        """Build the system prompt for patch selection."""
        candidate_count = candidate_count or 3  # Default to 3 if not provided
        return f"""You are an expert code evaluator. Given a codebase, a GitHub issue, and **{candidate_count} candidate patches** proposed by different sources, your responsibility is to **select the correct one** to solve the issue.

**CONTAINER TESTING CAPABILITIES:**
You have access to a Docker container with the project at the issue commit. The selected patch will be automatically tested in this container to validate its correctness. This provides real-world validation of the patch's effectiveness.

# WORK PROCESS:
You are given a software issue and multiple candidate patches. Your goal is to identify the patch that correctly resolves the issue.

Follow these steps methodically:

**1. Understand the Issue and Codebase**
Carefully read the issue description to comprehend the problem. You may need to examine the codebase for context, including:
    (1) Code referenced in the issue description;
    (2) The original code modified by each patch;
    (3) Unchanged parts of the same file;
    (4) Related files, functions, or modules that interact with the affected code.

**2. Analyze the Candidate Patches**
For each patch, analyze its logic and intended fix. Consider whether the changes align with the issue description and coding conventions.

**3. Validate Functionality (Optional but Recommended)**
If needed, write and run unit tests to evaluate the correctness and potential side effects of each patch.

**4. Select the Best Patch**
Choose the patch that best resolves the issue with minimal risk of introducing new problems.

# FINAL REPORT: If you have successfully selected the correct patch, submit your answer in the following format:
### Status: succeed
### Result: Patch-x
### Analysis: [Explain why Patch-x is correct.]

# IMPORTANT TIPS:
1. Never avoid making a selection.
2. Do not propose new patches.
3. There must be at least one correct patch.
4. Consider the confidence scores and regression test results when available.
"""
    
    def build_user_prompt(self, issue_description: str, project_path: str, candidate_patches: List[CandidatePatch], context_files: Optional[List[str]] = None, **kwargs) -> str:
        """Build the user prompt for patch selection."""
        user_prompt = f"""
[Codebase path]:
{project_path}

[GitHub issue description]:
```
{issue_description}
```

[Candidate Patches]:"""
        
        for idx, patch in enumerate(candidate_patches):
            user_prompt += f"\nPatch-{idx + 1}:\n```\n{patch.patch}\n```"
            
            # Add metadata if available
            if patch.confidence > 0:
                user_prompt += f"\nConfidence: {patch.confidence:.2f}"
            if not patch.is_success_regression:
                user_prompt += f"\nRegression Warning: This patch may cause regressions"
        
        # Add context files if provided
        if context_files:
            user_prompt += f"\n\n[Relevant Context Files]:\n"
            for file_path in context_files:
                user_prompt += f"- {file_path}\n"
        
        return user_prompt
    
    def _select_best_patch(
        self,
        issue_description: str,
        project_path: str,
        candidate_patches: List[CandidatePatch],
        context_files: Optional[List[str]] = None
    ) -> Tuple[int, str, Dict[str, Any]]:
        """
        Internal method to select the best patch from candidates.
        
        Args:
            issue_description: Description of the issue to be fixed
            project_path: Path to the project root
            candidate_patches: List of candidate patches
            context_files: Optional list of relevant file paths for context
            
        Returns:
            Tuple of (selected_patch_id, selected_patch_content, metadata)
        """
        if not candidate_patches:
            raise ValueError("No candidate patches provided")
        
        # Filter out empty patches
        valid_patches = [p for p in candidate_patches if p.patch.strip()]
        if not valid_patches:
            # If all patches are empty, return the first one
            return candidate_patches[0].id, candidate_patches[0].patch, {"reason": "no_valid_patches"}
        
        # Build the selection prompt
        system_prompt = self.build_system_prompt(len(valid_patches))
        
        user_prompt = f"""
[Codebase path]:
{project_path}

[GitHub issue description]:
```
{issue_description}
```

[Candidate Patches]:"""
        
        for idx, patch in enumerate(valid_patches):
            user_prompt += f"\nPatch-{idx + 1}:\n```\n{patch.patch}\n```"
            
            # Add metadata if available
            if patch.confidence > 0:
                user_prompt += f"\nConfidence: {patch.confidence:.2f}"
            if not patch.is_success_regression:
                user_prompt += f"\nRegression Warning: This patch may cause regressions"
        
        # Add context files if provided
        if context_files:
            user_prompt += f"\n\n[Relevant Context Files]:\n"
            for file_path in context_files:
                user_prompt += f"- {file_path}\n"
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Get LLM response
        response = self.llm_manager.chat(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            tools=[tool.json_definition() for tool in self.tools] if self.tools else None
        )
        
        if not response:
            if self.logger:
                self.logger.error("LLM response was empty")
            return valid_patches[0].id, valid_patches[0].patch, {"reason": "llm_error"}
        
        # Parse the response to extract selection
        selected_id, analysis = self._parse_selection_response(response, len(valid_patches))
        
        if selected_id is None:
            if self.logger:
                self.logger.warning("Could not parse selection from LLM response, using first patch")
            return valid_patches[0].id, valid_patches[0].patch, {"reason": "parse_error", "analysis": response}
        
        # Get the selected patch
        selected_patch = valid_patches[selected_id - 1]  # Convert to 0-based index
        
        # Test the selected patch in a new container
        test_results = {}
        executor = None
        try:
            # Create a new executor for this specific patch test
            executor = Executor(self.image_name)
            session_id = "0"  # Use default session
            
            # Apply the selected patch
            patch_success, patch_output = self.apply_patch_in_container(selected_patch.patch, executor, session_id)
            test_results["patch_applied"] = patch_success
            test_results["patch_output"] = patch_output
            
            if patch_success:
                # Run tests to validate the patch
                test_success, test_output = self.run_tests_in_container(executor, session_id)
                test_results["tests_passed"] = test_success
                test_results["test_output"] = test_output
            else:
                test_results["tests_passed"] = False
                test_results["test_output"] = "Patch application failed"
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during container testing: {e}")
            test_results["error"] = str(e)
        finally:
            # Always cleanup the executor
            if executor:
                try:
                    executor.shutdown()
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error during executor cleanup: {e}")
        
        metadata = {
            "selected_patch_id": selected_patch.id,
            "analysis": analysis,
            "confidence": selected_patch.confidence,
            "is_success_regression": selected_patch.is_success_regression,
            "total_candidates": len(valid_patches),
            "container_testing": test_results
        }
        
        return selected_patch.id, selected_patch.patch, metadata
    
    
    
    
    
    def select_patch(
        self,
        issue_description: str,
        project_path: str,
        candidate_patches: List[CandidatePatch],
        num_votes: int = 3,
        context_files: Optional[List[str]] = None
    ) -> Tuple[int, str, Dict[str, Any]]:
        """
        Select the best patch using intelligent voting.
        
        Uses multiple LLM evaluations to build consensus and select the most reliable patch.
        Supports both quick selection (num_votes=1) and careful consensus (num_votes>1).
        
        Args:
            issue_description: Description of the issue to be fixed
            project_path: Path to the project root
            candidate_patches: List of candidate patches
            num_votes: Number of voting rounds (1=quick, 3+=careful consensus)
            context_files: Optional list of relevant file paths for context
            
        Returns:
            Tuple of (selected_patch_id, selected_patch_content, metadata)
        """
        if not candidate_patches:
            raise ValueError("No candidate patches provided")
        
        votes = []
        analyses = []
        
        for vote_round in range(num_votes):
            if self.logger:
                self.logger.info(f"Conducting vote {vote_round + 1}/{num_votes}")
            
            try:
                selected_id, selected_patch, metadata = self._select_best_patch(
                    issue_description=issue_description,
                    project_path=project_path,
                    candidate_patches=candidate_patches,
                    context_files=context_files
                )
                
                votes.append(selected_id)
                analyses.append(metadata.get("analysis", ""))
                
                # Early termination if we have a clear majority
                if vote_round >= 2:  # After at least 3 votes
                    vote_counts = Counter(votes)
                    max_count = max(vote_counts.values())
                    if max_count > (vote_round + 1) / 2:  # Clear majority
                        break
                        
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error in vote {vote_round + 1}: {e}")
                continue
        
        if not votes:
            if self.logger:
                self.logger.error("All voting rounds failed, using first patch")
            return candidate_patches[0].id, candidate_patches[0].patch, {"reason": "voting_failed"}
        
        # Determine majority winner
        vote_counts = Counter(votes)
        most_common_votes = vote_counts.most_common()
        
        if not most_common_votes:
            return candidate_patches[0].id, candidate_patches[0].patch, {"reason": "no_votes"}
        
        # Get the most voted patch
        selected_id = most_common_votes[0][0]
        vote_count = most_common_votes[0][1]
        
        # Find the selected patch
        selected_patch = next((p for p in candidate_patches if p.id == selected_id), candidate_patches[0])
        
        metadata = {
            "selected_patch_id": selected_id,
            "vote_count": vote_count,
            "total_votes": len(votes),
            "vote_distribution": dict(vote_counts),
            "analyses": analyses,
            "confidence": selected_patch.confidence,
            "is_success_regression": selected_patch.is_success_regression
        }
        
        if self.logger:
            self.logger.info(f"Majority voting completed: Patch {selected_id} won with {vote_count}/{len(votes)} votes")
        
        return selected_id, selected_patch.patch, metadata
    


