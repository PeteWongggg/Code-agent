import sys
import json
sys.path.append('/Users/hengzhi/Code-agent')

from bash_tool import BashTool
from edit_tool import TextEditorTool

def print_tool_definitions():    
    print("=" * 80)
    print("BASH TOOL JSON DEFINITION")
    print("=" * 80)
    
    bash_tool = BashTool(model_provider="openrouter")
    bash_definition = bash_tool.json_definition()
    print(json.dumps(bash_definition, indent=2))
    
    print("\n" + "=" * 80)
    print("EDIT TOOL JSON DEFINITION")
    print("=" * 80)
    
    edit_tool = TextEditorTool(model_provider="openrouter")
    edit_definition = edit_tool.json_definition()
    print(json.dumps(edit_definition, indent=2))
    
    print("\n" + "=" * 80)
    print("COMPARISON: OPENAI vs ANTHROPIC MODEL PROVIDERS")
    print("=" * 80)
    
    # Show difference between model providers
    print("\nBashTool with OpenAI provider:")
    bash_openai = BashTool(model_provider="openai")
    print(json.dumps(bash_openai.json_definition(), indent=2))
    
    print("\nBashTool with Anthropic provider:")
    bash_anthropic = BashTool(model_provider="anthropic")
    print(json.dumps(bash_anthropic.json_definition(), indent=2))

if __name__ == "__main__":
    print_tool_definitions()
