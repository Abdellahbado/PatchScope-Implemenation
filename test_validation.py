#!/usr/bin/env python3
"""
Quick validation script to ensure the updated PatchScope implementation works correctly.
Tests imports and basic functionality without running actual experiments.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("🔍 Testing imports...")
    
    try:
        from config import PROMPTS, LAYER_CONFIG, MODEL_CONFIG, EXPERIMENT_CONFIG
        print("✅ config.py - imported successfully")
        
        # Check that the new template types are available
        template_categories = PROMPTS.get("template_categories", {})
        expected_types = ["patchscope", "few_shot", "minimal", "contextual", "standard"]
        
        for template_type in expected_types:
            if template_type in template_categories:
                print(f"   ✅ {template_type} templates available")
            else:
                print(f"   ⚠️  {template_type} templates missing")
                
        print(f"   📊 Available source prompts: {len(PROMPTS['source_prompts'])}")
        print(f"   📊 PatchScope templates: {len(PROMPTS['patchscope_templates'])}")
        print(f"   📊 Few-shot templates: {len(PROMPTS['few_shot_templates'])}")
        
    except ImportError as e:
        print(f"❌ config.py - import failed: {e}")
        return False
    
    try:
        from model_loader import ModelLoader
        print("✅ model_loader.py - imported successfully")
    except ImportError as e:
        print(f"❌ model_loader.py - import failed: {e}")
        return False
    
    try:
        from experiment_runner import ExperimentRunner
        print("✅ experiment_runner.py - imported successfully")
    except ImportError as e:
        print(f"❌ experiment_runner.py - import failed: {e}")
        return False
    
    try:
        from main import (
            run_quick_test, 
            run_patchscope_study, 
            run_multi_template_study,
            run_targeted_study
        )
        print("✅ main.py - imported successfully")
        print("   ✅ run_patchscope_study available")
        print("   ✅ run_multi_template_study available")
    except ImportError as e:
        print(f"❌ main.py - import failed: {e}")
        return False
    
    return True

def test_template_structure():
    """Test that template structure is properly configured."""
    print("\n🧪 Testing template structure...")
    
    from config import PROMPTS
    
    # Test template categories
    template_categories = PROMPTS.get("template_categories", {})
    
    for category, template_list_name in template_categories.items():
        if template_list_name in PROMPTS:
            templates = PROMPTS[template_list_name]
            print(f"   ✅ {category}: {len(templates)} templates")
            
            # Check that templates contain the marker 'x:'
            valid_templates = sum(1 for t in templates if 'x:' in t)
            print(f"      {valid_templates}/{len(templates)} templates have 'x:' marker")
        else:
            print(f"   ❌ {category}: template list '{template_list_name}' not found")

def test_few_shot_patterns():
    """Test that few-shot templates have proper patterns."""
    print("\n🎯 Testing few-shot patterns...")
    
    from config import PROMPTS
    
    few_shot_templates = PROMPTS.get("few_shot_templates", [])
    
    for i, template in enumerate(few_shot_templates):
        # Count the number of examples (sentences ending with periods before 'x:')
        parts = template.split('x:')[0].strip()
        sentences = [s.strip() for s in parts.split('.') if s.strip()]
        
        print(f"   Template {i+1}: {len(sentences)} examples")
        if len(sentences) >= 2:
            print(f"      ✅ Good few-shot pattern")
        else:
            print(f"      ⚠️  Could use more examples")

def main():
    """Run all validation tests."""
    print("🚀 PATCHSCOPE VALIDATION TESTS")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed!")
        return False
    
    # Test template structure
    test_template_structure()
    
    # Test few-shot patterns
    test_few_shot_patterns()
    
    print("\n🎉 All validation tests completed!")
    print("\n💡 Next steps:")
    print("1. Run: python main.py --mode quick")
    print("2. Run: python main.py --mode patchscope --template-type few_shot")
    print("3. Run: python main.py --mode templates")
    
    return True

if __name__ == "__main__":
    main()
