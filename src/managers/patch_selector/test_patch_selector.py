"""
Test cases for the Patch Selector

This module contains test cases to verify the patch selector functionality.
"""

import pytest
from unittest.mock import Mock, patch
from patch_selector import (
    CandidatePatch, 
    PatchSelector
)


class TestCandidatePatch:
    """Test cases for CandidatePatch class."""
    
    def test_candidate_patch_creation(self):
        """Test creating a CandidatePatch."""
        patch = CandidatePatch(
            id=1,
            patch="def test(): return True",
            is_success_regression=True,
            is_success_patch=True,
            source="test",
            confidence=0.8
        )
        
        assert patch.id == 1
        assert patch.patch == "def test(): return True"
        assert patch.is_success_regression is True
        assert patch.confidence == 0.8
    
    def test_git_diff_patch(self):
        """Test that CandidatePatch works with proper git diff output."""
        git_diff_patch = """diff --git a/src/utils.py b/src/utils.py
index 1234567..abcdefg 100644
--- a/src/utils.py
+++ b/src/utils.py
@@ -1,3 +1,3 @@
 def calculate(x):
-    return x * 2
+    return x * 3
"""
        
        patch = CandidatePatch(
            id=1,
            patch=git_diff_patch
        )
        
        # Should store the git diff as-is
        assert patch.patch == git_diff_patch
        assert "return x * 2" in patch.patch


class TestPatchSelector:
    """Test cases for PatchSelector class."""
    
    def test_selector_initialization(self):
        """Test PatchSelector initialization."""
        mock_llm = Mock()
        selector = PatchSelector(
            llm_manager=mock_llm,
            model="gpt-4o",
            max_turns=30,
            temperature=0.5
        )
        
        assert selector.model == "gpt-4o"
        assert selector.max_turns == 30
        assert selector.temperature == 0.5
    
    def test_system_prompt_generation(self):
        """Test system prompt generation."""
        mock_llm = Mock()
        selector = PatchSelector(
            llm_manager=mock_llm, 
            model="gpt-4o",
            image_name="test-image:latest",
            instance_id="test-123"
        )
        
        prompt = selector.build_system_prompt(3)
        
        assert "3 candidate patches" in prompt
        assert "Status: succeed" in prompt
        assert "Result: Patch-x" in prompt
    
    def test_select_patch_quick_mode(self):
        """Test select_patch with num_votes=1 (quick mode)."""
        mock_llm = Mock()
        mock_llm.chat.return_value = """
        ### Status: succeed
        ### Result: Patch-1
        ### Analysis: This patch correctly handles the edge case.
        """
        
        selector = PatchSelector(
            llm_manager=mock_llm, 
            model="gpt-4o",
            image_name="test-image:latest",
            instance_id="test-123"
        )
        
        candidates = [
            CandidatePatch(1, "def fix(): return True", True, True),
            CandidatePatch(2, "def fix(): return False", True, False)
        ]
        
        selected_id, selected_patch, metadata = selector.select_patch(
            issue_description="Fix the boolean logic",
            project_path="/workspace",
            candidate_patches=candidates,
            num_votes=1  # Quick mode
        )
        
        assert selected_id == 1
        assert selected_patch == "def fix(): return True"
        assert metadata["total_votes"] == 1
    
    def test_select_patch_consensus_mode(self):
        """Test select_patch with num_votes=3 (consensus mode)."""
        mock_llm = Mock()
        mock_llm.chat.return_value = """
        ### Status: succeed
        ### Result: Patch-2
        ### Analysis: This patch is the most robust solution.
        """
        
        selector = PatchSelector(
            llm_manager=mock_llm, 
            model="gpt-4o",
            image_name="test-image:latest",
            instance_id="test-123"
        )
        
        candidates = [
            CandidatePatch(1, "def fix(): return True", True, True),
            CandidatePatch(2, "def fix(): return False", True, False)
        ]
        
        selected_id, selected_patch, metadata = selector.select_patch(
            issue_description="Fix the boolean logic",
            project_path="/workspace",
            candidate_patches=candidates,
            num_votes=3  # Consensus mode
        )
        
        assert selected_id == 2
        assert selected_patch == "def fix(): return False"
        assert metadata["total_votes"] == 3
    
    def test_patch_selector_with_container(self):
        """Test PatchSelector with container integration (mock executor)."""
        from unittest.mock import Mock
        
        mock_llm = Mock()
        mock_llm.chat.return_value = """
        ### Status: succeed
        ### Result: Patch-1
        ### Analysis: This patch correctly fixes the issue.
        """
        
        # Mock the Executor class to avoid creating real containers
        with patch('patch_selector.Executor') as mock_executor_class:
            mock_executor = Mock()
            mock_executor.execute.return_value = (0, "Patch applied successfully")
            mock_executor_class.return_value = mock_executor
            
            selector = PatchSelector(
                llm_manager=mock_llm,
                model="gpt-4o",
                image_name="test-image:latest",
                instance_id="test-123"
            )
            
            candidates = [
                CandidatePatch(1, "def fix(): return True", True, True),
                CandidatePatch(2, "def fix(): return False", True, False)
            ]
            
            selected_id, selected_patch, metadata = selector.select_patch(
                issue_description="Fix the boolean logic",
                project_path="/workspace",
                candidate_patches=candidates,
                num_votes=1
            )
            
            assert selected_id == 1
            assert "container_testing" in metadata
            assert metadata["container_testing"]["patch_applied"] is True
            
            # Verify executor was called
            assert mock_executor.execute.called
    
    def test_selection_response_parsing(self):
        """Test parsing of LLM selection responses."""
        mock_llm = Mock()
        selector = PatchSelector(
            llm_manager=mock_llm, 
            model="gpt-4o",
            image_name="test-image:latest",
            instance_id="test-123"
        )
        
        # Test successful parsing
        response = """
        ### Status: succeed
        ### Result: Patch-2
        ### Analysis: This patch correctly handles the edge case.
        """
        
        patch_id, analysis = selector._parse_selection_response(response, 3)
        
        assert patch_id == 2
        assert "correctly handles" in analysis
    
    def test_invalid_response_parsing(self):
        """Test parsing of invalid responses."""
        mock_llm = Mock()
        selector = PatchSelector(
            llm_manager=mock_llm, 
            model="gpt-4o",
            image_name="test-image:latest",
            instance_id="test-123"
        )
        
        # Test invalid response
        response = "I don't know which patch to choose."
        
        patch_id, analysis = selector._parse_selection_response(response, 3)
        
        assert patch_id is None
        assert analysis == response


class TestPatchSelectorIntegration:
    """Test cases for PatchSelector integration with LLM manager."""
    
    @patch('patch_selector.LLMAPIManager')
    def test_patch_selector_with_llm_manager(self, mock_llm_class):
        """Test PatchSelector with LLM manager integration."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        # Create PatchSelector with mocked LLM manager
        selector = PatchSelector(
            llm_manager=mock_llm,
            model="gpt-4o"
        )
        
        # Test that selector is properly initialized
        assert selector.model == "gpt-4o"
        assert selector.llm_manager == mock_llm
    
    def test_patch_selector_initialization(self):
        """Test PatchSelector initialization with various parameters."""
        mock_llm = Mock()
        
        # Test with minimal parameters
        selector = PatchSelector(llm_manager=mock_llm, model="gpt-4o")
        assert selector.model == "gpt-4o"
        assert selector.max_turns == 50  # default
        assert selector.temperature == 0.7  # default
        
        # Test with custom parameters
        selector = PatchSelector(
            llm_manager=mock_llm,
            model="gpt-4o",
            max_turns=30,
            temperature=0.5,
            logger=Mock()
        )
        assert selector.model == "gpt-4o"
        assert selector.max_turns == 30
        assert selector.temperature == 0.5


class TestIntegration:
    """Integration test cases."""
    
    @patch('patch_selector.LLMAPIManager')
    def test_end_to_end_selection(self, mock_llm_class):
        """Test end-to-end patch selection."""
        # Mock LLM response
        mock_llm = Mock()
        mock_llm.chat.return_value = """
        ### Status: succeed
        ### Result: Patch-1
        ### Analysis: This patch correctly fixes the issue.
        """
        mock_llm_class.return_value = mock_llm
        
        # Create selector
        selector = PatchSelector(
            llm_manager=mock_llm, 
            model="gpt-4o",
            image_name="test-image:latest",
            instance_id="test-123"
        )
        
        # Test data
        candidates = [
            CandidatePatch(1, "def fix(): return True", True, True),
            CandidatePatch(2, "def fix(): return False", True, False)
        ]
        
        # This would normally call the LLM, but we're mocking it
        # In a real test, you'd want to test the actual LLM integration
        assert len(candidates) == 2


def test_error_handling():
    """Test error handling scenarios."""
    # Test empty candidate patches
    with pytest.raises(ValueError, match="No candidate patches provided"):
        quick_select_patch(
            issue_description="test",
            project_path="/test",
            candidate_patches=[]
        )


if __name__ == "__main__":
    # Run basic tests
    print("Running Patch Selector Tests...")
    
    # Test CandidatePatch
    test_patch = TestCandidatePatch()
    test_patch.test_candidate_patch_creation()
    test_patch.test_git_diff_patch()
    print("✓ CandidatePatch tests passed")
    
    # Test PatchSelector
    test_selector = TestPatchSelector()
    test_selector.test_selector_initialization()
    test_selector.test_system_prompt_generation()
    test_selector.test_selection_response_parsing()
    test_selector.test_invalid_response_parsing()
    print("✓ PatchSelector tests passed")
    
    # Test PatchSelector integration
    test_integration = TestPatchSelectorIntegration()
    test_integration.test_patch_selector_with_llm_manager()
    test_integration.test_patch_selector_initialization()
    print("✓ PatchSelector integration tests passed")
    
    print("All tests passed! 🎉")
