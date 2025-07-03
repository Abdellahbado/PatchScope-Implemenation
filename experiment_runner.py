"""
Experiment runner for PatchScope studies.
Orchestrates different types of experiments and manages execution flow.
"""

import random
import time
from typing import List, Dict, Any, Optional

from config import PROMPTS, LAYER_CONFIG, EXPERIMENT_CONFIG
from patchscope_core import PatchScope
from analysis import PatchScopeAnalyzer
from logger import ExperimentLogger


class ExperimentRunner:
    """Orchestrates PatchScope experiments."""
    
    def __init__(self, patchscope: PatchScope):
        self.patchscope = patchscope
        self.analyzer = PatchScopeAnalyzer()
        self.logger = None  # Will be initialized per experiment
        
    def run_patchscope_experiment(self, source_prompt: str, 
                                template_type: str = "patchscope",
                                num_templates: int = 3) -> List[Dict[str, Any]]:
        """
        Run PatchScope experiments with different template types.
        
        Args:
            source_prompt: Source entity to extract knowledge from
            template_type: Type of templates to use ("patchscope", "few_shot", "minimal", "contextual")
            num_templates: Number of templates to test
            
        Returns:
            List of experiment results
        """
        # Initialize logger for this experiment
        self.logger = ExperimentLogger(f"patchscope_{template_type}_{source_prompt.replace(' ', '_')}")
        
        # Get the appropriate template category
        template_category = PROMPTS["template_categories"].get(template_type, "patchscope_templates")
        template_list_name = template_category
        
        if template_list_name not in PROMPTS:
            print(f"⚠️  Template type '{template_type}' not found, using standard templates")
            template_list_name = "target_templates"
        
        # Select templates
        available_templates = PROMPTS[template_list_name]
        templates = random.sample(
            available_templates, 
            min(num_templates, len(available_templates))
        )
        
        # Log experiment start
        self.logger.log_experiment_start(
            f"PATCHSCOPE_{template_type.upper()}", 
            source_prompt, 
            f"{len(templates)} {template_type} templates",
            template_type=template_type,
            templates=templates,
            targeted_layers=LAYER_CONFIG["targeted_layers"]
        )
        
        all_results = []
        
        for i, template in enumerate(templates):
            print(f"\n--- {template_type.title()} Template {i+1}/{len(templates)} ---")
            print(f"Template: '{template}'")
            
            # Test targeted layer combinations for this template
            template_results = []
            for extract_layer in LAYER_CONFIG["targeted_layers"]:
                for inject_layer in LAYER_CONFIG["targeted_layers"]:
                    
                    self.logger.log_single_patch_attempt(
                        extract_layer, inject_layer, source_prompt, template
                    )
                    
                    result = self.patchscope.run_single_patch(
                        source_prompt, template, extract_layer, inject_layer
                    )
                    
                    self.logger.log_single_patch_result(result, extract_layer, inject_layer)
                    
                    if result:
                        result['template_type'] = template_type
                        result['template_index'] = i
                        result['template_text'] = template
                        template_results.append(result)
                        all_results.append(result)
                    
                    time.sleep(0.05)  # Small delay
            
            # Quick analysis for this template
            if template_results:
                template_analysis = self.analyzer.analyze_results(template_results, source_prompt)
                strong_count = template_analysis['summary']['strong_match_count']
                print(f"Template {i+1} results: {strong_count} strong matches")
                
                # Show best result for this template
                if template_analysis["strong_matches"]:
                    best_result = template_analysis["strong_matches"][0]
                    print(f"  Best: E{best_result['extract_layer']}→I{best_result['inject_layer']}: '{best_result['new_tokens'][:40]}...'")
        
        # Final analysis
        if all_results:
            final_analysis = self.analyzer.analyze_results(all_results, source_prompt)
            self.logger.log_analysis_results(final_analysis, source_prompt)
            
            hotspots = self.analyzer.find_knowledge_hotspots(final_analysis)
            self.logger.log_hotspot_analysis(hotspots)
            
            # Template comparison
            self._compare_template_performance(all_results, template_type)
        
        # Save session
        self.logger.save_session()
        
        return all_results
    
    def run_multi_template_experiment(self, source_prompt: str,
                                    template_types: List[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run PatchScope experiments across multiple template types.
        
        Args:
            source_prompt: Source entity to test
            template_types: List of template types to test
            
        Returns:
            Dictionary mapping template types to their results
        """
        if template_types is None:
            template_types = ["patchscope", "few_shot", "contextual", "minimal"]
        
        # Initialize logger
        self.logger = ExperimentLogger(f"multi_template_{source_prompt.replace(' ', '_')}")
        
        self.logger.log_experiment_start(
            "MULTI_TEMPLATE",
            source_prompt,
            f"Testing {len(template_types)} template types",
            template_types=template_types
        )
        
        results_by_template = {}
        
        for template_type in template_types:
            print(f"\n🔄 Testing {template_type.upper()} templates...")
            
            # Run experiment for this template type (fewer templates per type)
            results = self.run_patchscope_experiment(
                source_prompt, 
                template_type, 
                num_templates=2  # Fewer templates when testing multiple types
            )
            
            results_by_template[template_type] = results
            
            # Quick summary
            if results:
                analysis = self.analyzer.analyze_results(results, source_prompt)
                summary = analysis["summary"]
                success_rate = summary["strong_match_count"] / max(summary["successful_patches"], 1) * 100
                print(f"  {template_type.title()}: {summary['strong_match_count']} strong matches ({success_rate:.1f}% success rate)")
        
        # Final comparison across template types
        self._compare_across_template_types(results_by_template, source_prompt)
        
        return results_by_template
    
    def _compare_template_performance(self, results: List[Dict[str, Any]], template_type: str):
        """Compare performance across different templates of the same type."""
        print(f"\n📊 TEMPLATE PERFORMANCE ANALYSIS ({template_type.upper()})")
        print("=" * 50)
        
        # Group by template
        by_template = {}
        for result in results:
            template_idx = result.get('template_index', 0)
            if template_idx not in by_template:
                by_template[template_idx] = []
            by_template[template_idx].append(result)
        
        # Analyze each template
        for template_idx, template_results in by_template.items():
            if template_results:
                analysis = self.analyzer.analyze_results(template_results, "template_analysis")
                summary = analysis["summary"]
                success_rate = summary["strong_match_count"] / max(summary["successful_patches"], 1) * 100
                
                template_text = template_results[0].get('template_text', f'Template {template_idx + 1}')
                print(f"Template {template_idx + 1}: {summary['strong_match_count']}/{summary['successful_patches']} strong matches ({success_rate:.1f}%)")
                print(f"  Text: '{template_text[:60]}...'")
                
                if analysis["strong_matches"]:
                    best = analysis["strong_matches"][0]
                    print(f"  Best result: E{best['extract_layer']}→I{best['inject_layer']}: '{best['new_tokens'][:30]}...'")
    
    def _compare_across_template_types(self, results_by_template: Dict[str, List[Dict[str, Any]]], 
                                     source_prompt: str):
        """Compare performance across different template types."""
        print(f"\n🏆 CROSS-TEMPLATE TYPE COMPARISON")
        print("=" * 50)
        
        template_performance = {}
        
        for template_type, results in results_by_template.items():
            if results:
                analysis = self.analyzer.analyze_results(results, source_prompt)
                summary = analysis["summary"]
                
                success_rate = summary["strong_match_count"] / max(summary["successful_patches"], 1) * 100
                
                template_performance[template_type] = {
                    "strong_matches": summary["strong_match_count"],
                    "total_attempts": summary["successful_patches"],
                    "success_rate": success_rate,
                    "best_results": analysis["strong_matches"][:3]  # Top 3
                }
        
        # Sort by success rate
        sorted_types = sorted(template_performance.items(), 
                            key=lambda x: x[1]["success_rate"], 
                            reverse=True)
        
        print(f"📈 Performance Ranking for '{source_prompt}':")
        for i, (template_type, performance) in enumerate(sorted_types, 1):
            print(f"{i}. {template_type.upper()}: {performance['success_rate']:.1f}% "
                  f"({performance['strong_matches']}/{performance['total_attempts']} strong matches)")
            
            # Show best result for this template type
            if performance["best_results"]:
                best = performance["best_results"][0]
                print(f"   Best: E{best['extract_layer']}→I{best['inject_layer']}: '{best['new_tokens'][:40]}...'")
        
        # Log the comparison
        if self.logger:
            self.logger.log_experiment_summary("MULTI_TEMPLATE_COMPARISON", 0, {
                "template_rankings": [(t, p["success_rate"]) for t, p in sorted_types],
                "best_template_type": sorted_types[0][0] if sorted_types else "none",
                "source_prompt": source_prompt
            })
        
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
        # Initialize logger if not already done
        if not self.logger:
            self.logger = ExperimentLogger(f"targeted_{source_prompt.replace(' ', '_')}")
        
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
        
        # Log experiment start
        self.logger.log_experiment_start(
            "TARGETED", 
            source_prompt, 
            target_prompt,
            layer_subset=layer_subset,
            test_layers=layers
        )
        
        # Test each layer as both extract and inject
        results = []
        total_combinations = len(layers) * len(layers)
        completed = 0
        
        for extract_layer in layers:
            for inject_layer in layers:
                completed += 1
                
                self.logger.log_single_patch_attempt(
                    extract_layer, inject_layer, source_prompt, target_prompt
                )
                
                result = self.patchscope.run_single_patch(
                    source_prompt, target_prompt, extract_layer, inject_layer
                )
                
                self.logger.log_single_patch_result(result, extract_layer, inject_layer)
                
                if result:
                    result['template_type'] = 'targeted'
                    results.append(result)
                
                # Progress update
                if completed % 5 == 0:
                    print(f"  Progress: {completed}/{total_combinations} combinations tested")
        
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
