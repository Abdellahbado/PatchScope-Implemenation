"""
Main execution script for PatchScope experiments.
This is the entry point for running various PatchScope studies.
"""

# Update the main.py file

import argparse
import sys
import time
from typing import Optional

from model_loader import ModelLoader
from patchscope_core import PatchScope
from experiment_runner import ExperimentRunner
from config import PROMPTS, MODEL_REGISTRY

def setup_experiment(model_name: str = None) -> tuple:
    """Initialize model, tokenizer, and experiment components."""
    print("🚀 PATCHSCOPE EXPERIMENT SETUP")
    print("=" * 40)
    
    # Load model and tokenizer with specified model
    model_loader = ModelLoader(model_name)
    model, tokenizer = model_loader.load_model_and_tokenizer()
    
    # Display model info
    model_info = model_loader.get_model_info()
    print(f"\nModel Info:")
    print(f"  Model ID: {model_loader.model_name or 'default'}")
    print(f"  Structure: {model_info.get('structure_type', 'Unknown')}")
    print(f"  Total layers: {model_info.get('total_layers', 'Unknown')}")
    print(f"  Device: {model_info.get('device', 'Unknown')}")
    
    # Initialize PatchScope and experiment runner
    patchscope = PatchScope(model, tokenizer, model_loader)
    runner = ExperimentRunner(patchscope)
    
    # Initialize logger for setup
    from logger import ExperimentLogger
    logger = ExperimentLogger("setup")
    logger.log_model_info(model_info)
    
    return model_loader, patchscope, runner



def run_quick_test():
    """Run a quick test to verify everything is working."""
    print("\n🧪 QUICK FUNCTIONALITY TEST")
    print("=" * 30)
    
    model_loader, patchscope, runner = setup_experiment()
    
    # Initialize logger for quick test
    from logger import ExperimentLogger
    logger = ExperimentLogger("quick_test")
    
    # Simple test
    source_prompt = "George Washington"
    target_prompt = PROMPTS["target_templates"][0]
    
    logger.log_experiment_start("QUICK_TEST", source_prompt, target_prompt)
    
    print(f"Testing: '{source_prompt}' → '{target_prompt[:50]}...'")
    
    # Test a few strategic layer combinations
    test_combinations = [(2, 2), (7, 14), (14, 21), (21, 26)]
    
    successful_tests = 0
    total_tests = len(test_combinations)
    
    for extract_layer, inject_layer in test_combinations:
        logger.log_single_patch_attempt(extract_layer, inject_layer, source_prompt, target_prompt)
        
        result = patchscope.run_single_patch(
            source_prompt, target_prompt, extract_layer, inject_layer
        )
        
        logger.log_single_patch_result(result, extract_layer, inject_layer)
        
        if result and result['patch_applied']:
            successful_tests += 1
    
    # Summary
    print(f"\n📊 Quick Test Summary:")
    print(f"  Success rate: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")
    
    logger.log_experiment_summary("QUICK_TEST", 0, {
        "success_rate": f"{successful_tests}/{total_tests}",
        "successful_patches": successful_tests
    })
    logger.save_session()
    
    print("Quick test complete!")


def run_patchscope_study(source_prompt: Optional[str] = None, template_type: str = "patchscope"):
    """Run PatchScope experiments with different template types."""
    model_loader, patchscope, runner = setup_experiment()
    
    if source_prompt is None:
        source_prompt = "George Washington"
    
    start_time = time.time()
    
    # Run PatchScope experiment with specified template type
    results = runner.run_patchscope_experiment(source_prompt, template_type, num_templates=3)
    
    # Analyze results
    if results:
        analysis = runner.analyzer.analyze_results(results, source_prompt)
        runner.analyzer.print_analysis_summary(analysis, source_prompt)
        
        # Show hotspots
        hotspots = runner.analyzer.find_knowledge_hotspots(analysis)
        runner.analyzer.print_hotspot_analysis(hotspots)
        
        # Log final summary
        end_time = time.time()
        runner.logger.log_experiment_summary("PATCHSCOPE", end_time - start_time, {
            "total_results": len(results),
            "strong_matches": len(analysis["strong_matches"]),
            "template_type": template_type
        })


def run_multi_template_study(source_prompt: Optional[str] = None):
    """Run PatchScope experiments across multiple template types."""
    model_loader, patchscope, runner = setup_experiment()
    
    if source_prompt is None:
        source_prompt = "Albert Einstein"
    
    start_time = time.time()
    
    # Run multi-template experiment
    results_by_template = runner.run_multi_template_experiment(source_prompt)
    
    # Final analysis will be handled by the experiment runner
    
    # Log final summary  
    end_time = time.time()
    if runner.logger:
        runner.logger.log_experiment_summary("MULTI_TEMPLATE", end_time - start_time, {
            "template_types": list(results_by_template.keys()),
            "total_results": sum(len(results) for results in results_by_template.values())
        })
        runner.logger.save_session()


def run_targeted_study(source_prompt: Optional[str] = None):
    """Run a targeted study on specific prompts."""
    model_loader, patchscope, runner = setup_experiment()
    
    if source_prompt is None:
        source_prompt = "George Washington"
    
    target_prompt = PROMPTS["target_templates"][0]
    
    start_time = time.time()
    
    # Run targeted experiment
    results = runner.run_targeted_experiment(source_prompt, target_prompt)
    
    # Analyze results
    analysis = runner.analyzer.analyze_results(results, source_prompt)
    runner.analyzer.print_analysis_summary(analysis, source_prompt)
    
    # Show hotspots
    hotspots = runner.analyzer.find_knowledge_hotspots(analysis)
    runner.analyzer.print_hotspot_analysis(hotspots)
    
    # Log final summary
    if runner.logger:
        end_time = time.time()
        runner.logger.log_experiment_summary("TARGETED", end_time - start_time, {
            "total_results": len(results),
            "strong_matches": len(analysis["strong_matches"]),
            "target_template": "standard"
        })
        runner.logger.save_session()


def run_sweep_study(source_prompt: Optional[str] = None):
    """Run a comprehensive sweep study."""
    model_loader, patchscope, runner = setup_experiment()
    
    if source_prompt is None:
        source_prompt = "George Washington"
    
    target_prompt = PROMPTS["target_templates"][0]
    
    start_time = time.time()
    
    # Run sweep experiment
    results = runner.run_sweep_experiment(source_prompt, target_prompt)
    
    # Analyze results
    analysis = runner.analyzer.analyze_results(results, source_prompt)
    runner.analyzer.print_analysis_summary(analysis, source_prompt)
    
    # Show hotspots
    hotspots = runner.analyzer.find_knowledge_hotspots(analysis)
    runner.analyzer.print_hotspot_analysis(hotspots)
    
    # Log final summary
    if runner.logger:
        end_time = time.time()
        runner.logger.log_experiment_summary("SWEEP", end_time - start_time, {
            "total_results": len(results),
            "strong_matches": len(analysis["strong_matches"]),
            "experiment_type": "strategic_sweep"
        })
        runner.logger.save_session()


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
    parser = argparse.ArgumentParser(
        description="PatchScope Experiments - Advanced Interpretability Analysis",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Add model selection argument
    parser.add_argument(
        "--model", 
        choices=list(MODEL_REGISTRY.keys()),
        default=None,
        help=(
            "Model to use for experiments:\n" +
            "\n".join([f"  {name}: {config['model_id']}" for name, config in MODEL_REGISTRY.items()])
        )
    )
    
    parser.add_argument(
        "--mode", 
        choices=["quick", "targeted", "sweep", "comprehensive", "multi", "patchscope", "templates"],
        default="quick",
        help=(
            "Experiment mode to run:\n"
            "  quick:         Quick functionality test\n"
            "  targeted:      Test predefined layer combinations\n"
            "  sweep:         Strategic layer combination sweep\n"
            "  comprehensive: Full multi-phase study\n"
            "  multi:         Multi-prompt comparison\n"
            "  patchscope:    PatchScope experiments with specific template type\n"
            "  templates:     PatchScope across multiple template types"
        )
    )
    
    parser.add_argument(
        "--source-prompt",
        type=str,
        help="Source prompt to use (default varies by mode)"
    )
    
    parser.add_argument(
        "--template-type",
        choices=["patchscope", "few_shot", "minimal", "contextual", "standard"],
        default="patchscope",
        help="Template type for PatchScope experiments"
    )
    
    args = parser.parse_args()
    
    try:
        print(f"🎯 Starting PatchScope experiment: {args.mode.upper()}")
        if args.model:
            print(f"🤖 Using model: {args.model} ({MODEL_REGISTRY[args.model]['model_id']})")
        else:
            print(f"🤖 Using default model")
        print(f"⏰ Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if args.mode == "quick":
            run_quick_test(args.model)
        elif args.mode == "targeted":
            run_targeted_study(args.source_prompt, args.model)
        elif args.mode == "sweep":
            run_sweep_study(args.source_prompt, args.model)
        elif args.mode == "comprehensive":
            run_comprehensive_study(args.model)
        elif args.mode == "multi":
            run_multi_prompt_study(args.model)
        elif args.mode == "patchscope":
            run_patchscope_study(args.source_prompt, args.template_type, args.model)
        elif args.mode == "templates":
            run_multi_template_study(args.source_prompt, args.model)
        
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        sys.exit(1)



if __name__ == "__main__":
    # Enhanced default behavior with logging
    if len(sys.argv) == 1:
        print("🚀 Running quick test (use --help for more options)")
        print("📝 This will create log files in the current directory")
        run_quick_test()
    else:
        main()
