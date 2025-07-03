"""
Experiment runner for PatchScope studies.
Orchestrates different types of experiments and manages execution flow.
"""

import random
from typing import List, Dict, Any, Optional

from config import PROMPTS, LAYER_CONFIG, EXPERIMENT_CONFIG
from patchscope_core import PatchScope
from analysis import PatchScopeAnalyzer


class ExperimentRunner:
    """Orchestrates PatchScope experiments."""
    
    def __init__(self, patchscope: PatchScope):
        self.patchscope = patchscope
        self.analyzer = PatchScopeAnalyzer()
        
    def run_targeted_experiment(self, source_prompt: str, target_prompt: str,
                              layer_subset: str = "targeted") -> List[Dict[str, Any]]:
        """
        Run a targeted experiment with a predefined set of layer combinations.
        
        Args:
            source_prompt: Source prompt for extraction
            target_prompt: Target prompt for injection
            layer_subset: Which layer subset to use ("targeted", "early", "mid", "late")
            
        Returns:
            List of experiment results
        """
        if layer_subset == "targeted":
            layers = LAYER_CONFIG["targeted_layers"]
        elif layer_subset == "early":
            layers = LAYER_CONFIG["early_layers"][::2]  # Every 2nd layer
        elif layer_subset == "mid":
            layers = LAYER_CONFIG["mid_layers"][::2]
        elif layer_subset == "late":
            layers = LAYER_CONFIG["late_layers"][::2]
        else:
            layers = LAYER_CONFIG["targeted_layers"]
        
        print(f"🎯 TARGETED EXPERIMENT: {layer_subset.upper()} LAYERS")
        print(f"Source: '{source_prompt}'")
        print(f"Target: '{target_prompt}'")
        print(f"Testing layers: {layers}")
        
        # Test each layer as both extract and inject
        results = []
        for extract_layer in layers:
            for inject_layer in layers:
                result = self.patchscope.run_single_patch(
                    source_prompt, target_prompt, extract_layer, inject_layer
                )
                if result:
                    results.append(result)
        
        return results
    
    def run_sweep_experiment(self, source_prompt: str, target_prompt: str,
                           max_combinations: int = 50) -> List[Dict[str, Any]]:
        """
        Run a sweep experiment testing strategic layer combinations.
        
        Args:
            source_prompt: Source prompt
            target_prompt: Target prompt
            max_combinations: Maximum number of layer combinations to test
            
        Returns:
            List of experiment results
        """
        total_layers = LAYER_CONFIG["total_layers"]
        
        # Generate strategic layer combinations
        combinations = []
        
        # 1. Same layer (diagonal)
        step = max(1, total_layers // 10)
        for i in range(0, total_layers, step):
            combinations.append((i, i))
        
        # 2. Extract from early, inject to various
        early_layers = [2, 5, 8]
        for extract in early_layers:
            for inject in range(extract, min(extract + 15, total_layers), 3):
                combinations.append((extract, inject))
        
        # 3. Extract from mid, inject to various  
        mid_layers = [12, 16, 20]
        for extract in mid_layers:
            for inject in range(max(0, extract - 10), min(extract + 10, total_layers), 3):
                combinations.append((extract, inject))
        
        # 4. Extract from late, inject to earlier (knowledge refinement)
        late_layers = [22, 25]
        for extract in late_layers:
            for inject in range(max(0, extract - 15), extract, 3):
                combinations.append((extract, inject))
        
        # Remove duplicates and limit
        combinations = list(set(combinations))
        if len(combinations) > max_combinations:
            # Prioritize diverse combinations
            combinations = self._select_diverse_combinations(combinations, max_combinations)
        
        print(f"🔄 SWEEP EXPERIMENT")
        print(f"Source: '{source_prompt}'")
        print(f"Target: '{target_prompt}'")
        print(f"Testing {len(combinations)} layer combinations")
        
        # Run experiments
        results = []
        for i, (extract_layer, inject_layer) in enumerate(combinations):
            result = self.patchscope.run_single_patch(
                source_prompt, target_prompt, extract_layer, inject_layer
            )
            if result:
                results.append(result)
                
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(combinations)} combinations tested")
        
        return results
    
    def run_multi_prompt_experiment(self, num_prompts: int = 3, 
                                   experiment_type: str = "targeted") -> Dict[str, List[Dict[str, Any]]]:
        """
        Run experiments across multiple source prompts.
        
        Args:
            num_prompts: Number of source prompts to test
            experiment_type: Type of experiment ("targeted" or "sweep")
            
        Returns:
            Dictionary mapping source prompts to their results
        """
        # Select prompts
        source_prompts = random.sample(PROMPTS["source_prompts"], 
                                     min(num_prompts, len(PROMPTS["source_prompts"])))
        target_prompt = random.choice(PROMPTS["target_templates"])
        
        print(f"🔬 MULTI-PROMPT EXPERIMENT ({experiment_type.upper()})")
        print(f"Testing {len(source_prompts)} source prompts")
        print(f"Target template: '{target_prompt}'")
        
        results_by_prompt = {}
        
        for i, source_prompt in enumerate(source_prompts):
            print(f"\n--- Prompt {i+1}/{len(source_prompts)}: {source_prompt} ---")
            
            if experiment_type == "targeted":
                results = self.run_targeted_experiment(source_prompt, target_prompt)
            elif experiment_type == "sweep":
                results = self.run_sweep_experiment(source_prompt, target_prompt)
            else:
                results = self.run_targeted_experiment(source_prompt, target_prompt)
            
            results_by_prompt[source_prompt] = results
            
            # Quick analysis for this prompt
            analysis = self.analyzer.analyze_results(results, source_prompt)
            summary = analysis["summary"]
            print(f"Quick results: {summary['strong_match_count']} strong matches, "
                  f"{summary['partial_match_count']} partial matches")
        
        return results_by_prompt
    
    def run_comprehensive_study(self):
        """
        Run a comprehensive PatchScope study with multiple experiments.
        """
        print("🧪 COMPREHENSIVE PATCHSCOPE STUDY")
        print("=" * 50)
        
        # Phase 1: Quick targeted test
        print("\nPhase 1: Quick targeted test")
        source_prompt = "George Washington"
        target_prompt = PROMPTS["target_templates"][0]
        
        targeted_results = self.run_targeted_experiment(source_prompt, target_prompt)
        analysis = self.analyzer.analyze_results(targeted_results, source_prompt)
        self.analyzer.print_analysis_summary(analysis, source_prompt)
        
        # Phase 2: Focused sweep based on Phase 1 results
        print("\nPhase 2: Strategic sweep")
        hotspots = self.analyzer.find_knowledge_hotspots(analysis)
        self.analyzer.print_hotspot_analysis(hotspots)
        
        if analysis["strong_matches"]:
            sweep_results = self.run_sweep_experiment(source_prompt, target_prompt, max_combinations=30)
            sweep_analysis = self.analyzer.analyze_results(sweep_results, source_prompt)
            print("\nSweep results:")
            self.analyzer.print_analysis_summary(sweep_analysis, source_prompt)
        
        # Phase 3: Multi-prompt comparison
        print("\nPhase 3: Multi-prompt comparison")
        multi_results = self.run_multi_prompt_experiment(num_prompts=3, experiment_type="targeted")
        self.analyzer.compare_prompts(multi_results)
        
        print(f"\n{'='*60}")
        print("COMPREHENSIVE STUDY COMPLETE!")
        print("Key insights:")
        
        # Summarize key findings
        if analysis["strong_matches"]:
            best_result = max(analysis["strong_matches"], 
                            key=lambda x: len(x['new_tokens']))
            print(f"- Best single result: E{best_result['extract_layer']}→"
                  f"I{best_result['inject_layer']}: '{best_result['new_tokens']}'")
        
        if hotspots.get("best_layer_pairs"):
            best_pair = hotspots["best_layer_pairs"][0]
            (ext, inj), count = best_pair
            print(f"- Most consistent layer pair: E{ext}→I{inj} ({count} successes)")
    
    def _select_diverse_combinations(self, combinations: List[tuple], 
                                   max_count: int) -> List[tuple]:
        """Select diverse layer combinations to test."""
        # Sort by sum of layers to get spread across the layer space
        combinations.sort(key=lambda x: x[0] + x[1])
        
        # Select evenly spaced combinations
        step = len(combinations) / max_count
        selected = []
        for i in range(max_count):
            idx = int(i * step)
            if idx < len(combinations):
                selected.append(combinations[idx])
        
        return selected
