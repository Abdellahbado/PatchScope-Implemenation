"""
Demonstration script showing how to use the PatchScope implementation.
Run this to see the modular structure in action.
"""

import sys
import os

# Add current directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from model_loader import ModelLoader
    from patchscope_core import PatchScope
    from experiment_runner import ExperimentRunner
    from analysis import PatchScopeAnalyzer
    from config import PROMPTS, LAYER_CONFIG
    
    def demo_modular_usage():
        """Demonstrate how to use each module independently."""
        
        print("🎭 PATCHSCOPE MODULAR DEMO")
        print("=" * 40)
        
        # Step 1: Load model using ModelLoader
        print("\n1️⃣ Loading Model...")
        model_loader = ModelLoader()
        model, tokenizer = model_loader.load_model_and_tokenizer()
        
        # Display model info
        info = model_loader.get_model_info()
        print(f"   Loaded: {info['total_layers']} layer model")
        
        # Step 2: Initialize PatchScope core
        print("\n2️⃣ Initializing PatchScope...")
        patchscope = PatchScope(model, tokenizer, model_loader)
        
        # Step 3: Run a single patch experiment
        print("\n3️⃣ Single Patch Test...")
        source = "George Washington"
        target = PROMPTS["target_templates"][0]
        
        result = patchscope.run_single_patch(
            source_prompt=source,
            target_prompt=target,
            extract_layer=14,  # Middle layer
            inject_layer=21    # Later layer
        )
        
        if result:
            print(f"   Extract L14 → Inject L21:")
            print(f"   Generated: '{result['new_tokens']}'")
        
        # Step 4: Use the analyzer
        print("\n4️⃣ Analysis Demo...")
        analyzer = PatchScopeAnalyzer()
        
        # Create some dummy results for demo
        demo_results = [result] if result else []
        
        if demo_results:
            analysis = analyzer.analyze_results(demo_results, source)
            analyzer.print_analysis_summary(analysis, source)
        
        # Step 5: Show experiment runner
        print("\n5️⃣ Experiment Runner Demo...")
        runner = ExperimentRunner(patchscope)
        
        # Run a small targeted experiment
        targeted_results = runner.run_targeted_experiment(
            source_prompt=source,
            target_prompt=target,
            layer_subset="targeted"
        )
        
        print(f"   Completed {len(targeted_results)} targeted experiments")
        
        # Step 6: Show configuration flexibility
        print("\n6️⃣ Configuration Demo...")
        print(f"   Available source prompts: {len(PROMPTS['source_prompts'])}")
        print(f"   Available target templates: {len(PROMPTS['target_templates'])}")
        print(f"   Targeted layers: {LAYER_CONFIG['targeted_layers']}")
        
        print("\n✅ Demo complete! Each module can be used independently.")
        print("   Check main.py for full experiment modes.")
        
    if __name__ == "__main__":
        demo_modular_usage()
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have installed the requirements:")
    print("pip install torch transformers accelerate")
except Exception as e:
    print(f"❌ Demo error: {e}")
    print("This is a demonstration of the modular structure.")
    print("Some errors are expected if the model isn't available.")
