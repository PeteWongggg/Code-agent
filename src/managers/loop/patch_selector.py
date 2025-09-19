from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import re

from ..llm_api.api_manager import LLMAPIManager
from ..prompts.prompts_manager import PromptsManager
from ...tools.base import Tool
from ...tools.executor import Executor


@dataclass
class CandidatePatch:
    id: int
    patch: str


class PatchSelector:
    def __init__(
        self,
        llm_manager: LLMAPIManager,
        config: Dict[str, Any] | None = None,
        image_name: str,
        instance_id: str,
        tools: Optional[List[Tool]] = None,
        logger: Optional[Any] = None,
        prompts_manager: Optional[PromptsManager] = None
    ):
        self.llm_manager = llm_manager
        self.image_name = image_name
        self.instance_id = instance_id
        self.tools = tools
        self.logger = logger
        self.prompts_manager = prompts_manager or PromptsManager({})
        self.config = config or {}
    
    
    def build_user_prompt(self, issue_description: str, project_path: str, candidate_patches: List[CandidatePatch], context_files: Optional[List[str]] = None, **kwargs) -> str:
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
        
        system_prompt = self.prompts_manager.get_selector_system(len(candidate_patches))
        
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
        
        temperature = self.config.get("runner", {}).get("selector_loop", {}).get("temperature", 0.7)
        response = self.llm_manager.chat(
            messages=messages,
            tools=[tool.json_definition() for tool in self.tools] if self.tools else None
        )
        
        if not response:
            if self.logger:
                self.logger.error("LLM response was empty")
            return candidate_patches[0].id, candidate_patches[0].patch, {"reason": "llm_error"}
        
        # Parse the response to extract selection
        selected_id = self._parse_selection_response(response, len(candidate_patches))
        
        if selected_id is None:
            if self.logger:
                self.logger.warning("Could not parse selection from LLM response, using first patch")
            return candidate_patches[0].id, candidate_patches[0].patch, {"reason": "parse_error", "analysis": response}
        
        # Get the selected patch
        selected_patch = candidate_patches[selected_id - 1]  # Convert to 0-based index
        
        test_results = {}
        executor = None
        try:
            executor = Executor(self.image_name)
            session_id = "0"  

            patch_success, patch_output = self.apply_patch_in_container(selected_patch.patch, executor, session_id)
            test_results["patch_applied"] = patch_success
            test_results["patch_output"] = patch_output
            
            if patch_success:
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
            if executor:
                try:
                    executor.shutdown()
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error during executor cleanup: {e}")
        
        metadata = {
            "selected_patch_id": selected_patch.id,
            "total_candidates": len(candidate_patches),
            "container_testing": test_results
        }
        
        return selected_patch.id, selected_patch.patch, metadata
    
    def _parse_selection_response(self, response: str, num_patches: int) -> Tuple[Optional[int], str]:
        pattern = r'###(\d+)###'
        match = re.search(pattern, response)
        
        if match:
            try:
                patch_number = int(match.group(1))
                if 1 <= patch_number <= num_patches:
                    return patch_number
                else:
                    if self.logger:
                        self.logger.warning(f"Patch number {patch_number} out of range (1-{num_patches})")
                    return None
            except ValueError:
                if self.logger:
                    self.logger.warning(f"Could not parse patch number from: {match.group(1)}")
                return None
        else:
            if self.logger:
                self.logger.warning("No valid patch selection found in response")
            return None
    
    def select_patch(
        self,
        issue_description: str,
        project_path: str,
        candidate_patches: List[CandidatePatch],
        num_votes: int = 3,
        context_files: Optional[List[str]] = None
    ) -> Tuple[int, str, Dict[str, Any]]:
        
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
        }
        
        if self.logger:
            self.logger.info(f"Majority voting completed: Patch {selected_id} won with {vote_count}/{len(votes)} votes")
        
        return selected_id, selected_patch.patch, metadata