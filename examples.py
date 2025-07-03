#!/usr/bin/env python3
"""
Example usage of the enhanced PatchScope implementation.
Shows how to use different experiment types and logging features.
"""

from main import (
    run_quick_test, 
    run_patchscope_study, 
    run_multi_template_study,
    run_targeted_study
)

def demonstrate_features():
    """Demonstrate the key features of the enhanced PatchScope implementation."""
    
    print("🌟 PATCHSCOPE ENHANCED FEATURES DEMO")
    print("=" * 50)
    
    print("\n1️⃣ Quick Test (Basic functionality)")
    print("This will test a few layer combinations and create basic logs.")
    # Uncomment to run: run_quick_test()
    
    print("\n2️⃣ PatchScope with Different Template Types")
    print("Tests PatchScope with various prompt styles (patchscope, few-shot, contextual, minimal).")
    print("Example: 'Syria: Country in the Middle East. Leonardo DiCaprio: American actor. x:'")
    # Uncomment to run: run_patchscope_study("Albert Einstein", "patchscope")
    
    print("\n3️⃣ Multi-Template Experiment")
    print("Compares performance across all template types for comprehensive analysis.")
    print("Tests: patchscope, few_shot, contextual, and minimal templates")
    # Uncomment to run: run_multi_template_study("Marie Curie")
    
    print("\n4️⃣ Enhanced Logging Features")
    print("All experiments now create detailed logs:")
    print("  📄 experiment_history.log - Real-time event log")
    print("  📊 detailed_results.json - Complete results data")
    print("  📝 experiment_summary.txt - Human-readable summary")
    
    print("\n5️⃣ Enhanced Prompts and Analysis")
    print("Now supports:")
    print("  • PatchScope templates (mixed context)")
    print("  • Few-shot prompting patterns")
    print("  • Contextual templates (rich context)")
    print("  • Minimal context templates")
    print("  • Multi-factor match scoring")
    print("  • Template performance comparison")
    
    print("\n6️⃣ Command Line Usage Examples")
    print("python main.py --mode quick")
    print("python main.py --mode patchscope --source-prompt 'Napoleon' --template-type few_shot")
    print("python main.py --mode templates --source-prompt 'Darwin'")
    print("python main.py --mode comprehensive")
    print("python main.py --mode targeted --source-prompt 'Shakespeare'")
    
    print("\n🔍 Detailed Output Examples")
    print("The enhanced version provides much more detailed output:")
    print("• Layer-by-layer extraction details")
    print("• Patch application confirmation")
    print("• Representation norm changes")
    print("• Real-time progress tracking")
    print("• Success/failure analysis")
    
    print("\n💡 Tips for Usage")
    print("1. Start with 'quick' mode to verify everything works")
    print("2. Use 'patchscope' mode with different template types")
    print("3. Use 'templates' mode for comprehensive comparison")
    print("4. Check log files for detailed analysis")
    print("5. Logs use relative paths - work in any environment")


def show_log_structure():
    """Show what the log files contain."""
    print("\n📁 LOG FILE STRUCTURE")
    print("=" * 30)
    
    print("\n📄 experiment_history.log:")
    print("```")
    print("[2025-07-03T14:30:15] EXPERIMENT_START: Starting quick_test")
    print("[2025-07-03T14:30:16] MODEL_LOADED: Model: {'total_layers': 28, ...}")
    print("[2025-07-03T14:30:17] PATCH_SUCCESS: E14→I21: first president of the United States")
    print("[2025-07-03T14:30:18] ANALYSIS_COMPLETE: Analysis for George Washington")
    print("```")
    
    print("\n📊 detailed_results.json:")
    print("```json")
    print("{")
    print('  "session_id": "20250703_143015",')
    print('  "experiment_name": "patchscope_few_shot_George_Washington",')
    print('  "environment": "Local",')
    print('  "results": [')
    print('    {')
    print('      "extract_layer": 14,')
    print('      "inject_layer": 21,')
    print('      "new_tokens": "first president of the United States",')
    print('      "match_score": 0.85,')
    print('      "template_type": "few_shot"')
    print('    }')
    print('  ]')
    print('}')
    print("```")
    
    print("\n📝 experiment_summary.txt:")
    print("```")
    print("PatchScope Experiment Summary")
    print("========================================")
    print("Session ID: 20250703_143015")
    print("Environment: Local")
    print("Total patch attempts: 25")
    print("Successful patches: 18")
    print("Success rate: 72.0%")
    print("```")


if __name__ == "__main__":
    print("🎭 This is a demonstration of PatchScope features.")
    print("Uncomment the function calls in the code to run actual experiments.")
    
    demonstrate_features()
    show_log_structure()
    
    print(f"\n✨ Ready to run experiments!")
    print("Try: python main.py --mode entity --source-prompt 'George Washington'")
