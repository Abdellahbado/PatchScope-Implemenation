"""
Main execution script for PatchScope experiments.
This is the entry point for running various PatchScope studies.
"""

import sys
import argparse
from typing import Optional

from model_loader import ModelLoader
from patchscope_core import PatchScope
from experiment_runner import ExperimentRunner
from config import PROMPTS


def setup_experiment() -> tuple:
    """Initialize model, tokenizer, and experiment components."""
    print("🚀 PATCHSCOPE EXPERIMENT SETUP")
    print("=" * 40)
    
    # Load model and tokenizer
    model_loader = ModelLoader()
    model, tokenizer = model_loader.load_model_and_tokenizer()
    
    # Display model info
    model_info = model_loader.get_model_info()
    print(f"\nModel Info:")
    print(f"  Structure: {model_info.get('structure_type', 'Unknown')}")
    print(f"  Total layers: {model_info.get('total_layers', 'Unknown')}")
    print(f"  Device: {model_info.get('device', 'Unknown')}")
    
    # Initialize PatchScope and experiment runner
    patchscope = PatchScope(model, tokenizer, model_loader)
    runner = ExperimentRunner(patchscope)
    
    return model_loader, patchscope, runner


def run_quick_test():
    """Run a quick test to verify everything is working."""
    print("\n🧪 QUICK FUNCTIONALITY TEST")
    print("=" * 30)
    
    model_loader, patchscope, runner = setup_experiment()
    
    # Simple test
    source_prompt = "George Washington"
    target_prompt = PROMPTS["target_templates"][0]
    
    print(f"Testing: '{source_prompt}' → '{target_prompt}'")
    
    # Test a few strategic layer combinations
    test_combinations = [(2, 2), (7, 14), (14, 21), (21, 26)]
    
    for extract_layer, inject_layer in test_combinations:
        result = patchscope.run_single_patch(
            source_prompt, target_prompt, extract_layer, inject_layer
        )
        
        if result and result['patch_applied']:
            print(f"✅ E{extract_layer:2d}→I{inject_layer:2d}: {result['new_tokens'][:40]}...")
        else:
            print(f"❌ E{extract_layer:2d}→I{inject_layer:2d}: Failed")
    
    print("\nQuick test complete!")


def run_targeted_study(source_prompt: Optional[str] = None):
    """Run a targeted study on specific prompts."""
    model_loader, patchscope, runner = setup_experiment()
    
    if source_prompt is None:
        source_prompt = "George Washington"
    
    target_prompt = PROMPTS["target_templates"][0]
    
    # Run targeted experiment
    results = runner.run_targeted_experiment(source_prompt, target_prompt)
    
    # Analyze results
    analysis = runner.analyzer.analyze_results(results, source_prompt)
    runner.analyzer.print_analysis_summary(analysis, source_prompt)
    
    # Show hotspots
    hotspots = runner.analyzer.find_knowledge_hotspots(analysis)
    runner.analyzer.print_hotspot_analysis(hotspots)


def run_sweep_study(source_prompt: Optional[str] = None):
    """Run a comprehensive sweep study."""
    model_loader, patchscope, runner = setup_experiment()
    
    if source_prompt is None:
        source_prompt = "George Washington"
    
    target_prompt = PROMPTS["target_templates"][0]
    
    # Run sweep experiment
    results = runner.run_sweep_experiment(source_prompt, target_prompt)
    
    # Analyze results
    analysis = runner.analyzer.analyze_results(results, source_prompt)
    runner.analyzer.print_analysis_summary(analysis, source_prompt)
    
    # Show hotspots
    hotspots = runner.analyzer.find_knowledge_hotspots(analysis)
    runner.analyzer.print_hotspot_analysis(hotspots)


def run_comprehensive_study():
    """Run the full comprehensive study."""
    model_loader, patchscope, runner = setup_experiment()
    runner.run_comprehensive_study()


def run_multi_prompt_study():
    """Run multi-prompt comparison study."""
    model_loader, patchscope, runner = setup_experiment()
    
    # Run multi-prompt experiment
    results_by_prompt = runner.run_multi_prompt_experiment(num_prompts=4)
    
    # Final comparison
    print(f"\n🏁 FINAL MULTI-PROMPT ANALYSIS")
    print("=" * 40)
    runner.analyzer.compare_prompts(results_by_prompt)


def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(description="PatchScope Experiments")
    parser.add_argument(
        "--mode", 
        choices=["quick", "targeted", "sweep", "comprehensive", "multi"],
        default="quick",
        help="Experiment mode to run"
    )
    parser.add_argument(
        "--source-prompt",
        type=str,
        help="Source prompt to use (default: 'George Washington')"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "quick":
            run_quick_test()
        elif args.mode == "targeted":
            run_targeted_study(args.source_prompt)
        elif args.mode == "sweep":
            run_sweep_study(args.source_prompt)
        elif args.mode == "comprehensive":
            run_comprehensive_study()
        elif args.mode == "multi":
            run_multi_prompt_study()
        
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\nError during experiment: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # If no command line args, run quick test by default
    if len(sys.argv) == 1:
        print("Running quick test (use --help for more options)")
        run_quick_test()
    else:
        main()
