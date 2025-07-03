"""
Analysis and visualization utilities for PatchScope experiments.
"""

from typing import List, Dict, Any
from collections import defaultdict

from config import PROMPTS, ANALYSIS_CONFIG


class PatchScopeAnalyzer:
    """Analyzer for PatchScope experiment results."""
    
    def __init__(self):
        self.results = []
        
    def analyze_results(self, results: List[Dict[str, Any]], source_prompt: str) -> Dict[str, Any]:
        """
        Analyze experiment results to find knowledge patterns.
        
        Args:
            results: List of experiment results
            source_prompt: Source prompt used in experiments
            
        Returns:
            Analysis summary
        """
        if not results:
            return {"error": "No results to analyze"}
            
        # Get expected keywords for this source
        expected_keywords = PROMPTS["analysis_keywords"].get(
            source_prompt, 
            [source_prompt.lower()]
        )
        
        # Categorize results
        analysis = {
            "strong_matches": [],
            "partial_matches": [],
            "weak_matches": [],
            "failed_patches": []
        }
        
        for result in results:
            if not result.get('patch_applied', False):
                analysis["failed_patches"].append(result)
                continue
                
            new_text_lower = result['new_tokens'].lower()
            
            # Count keyword matches
            matches = sum(1 for keyword in expected_keywords if keyword in new_text_lower)
            
            # Categorize based on match count
            if matches >= ANALYSIS_CONFIG["match_thresholds"]["strong_match"]:
                analysis["strong_matches"].append(result)
            elif matches >= ANALYSIS_CONFIG["match_thresholds"]["partial_match"]:
                analysis["partial_matches"].append(result)
            else:
                analysis["weak_matches"].append(result)
        
        # Add summary statistics
        analysis["summary"] = {
            "total_experiments": len(results),
            "successful_patches": len(results) - len(analysis["failed_patches"]),
            "strong_match_count": len(analysis["strong_matches"]),
            "partial_match_count": len(analysis["partial_matches"]),
            "weak_match_count": len(analysis["weak_matches"]),
            "expected_keywords": expected_keywords
        }
        
        return analysis
    
    def print_analysis_summary(self, analysis: Dict[str, Any], source_prompt: str):
        """Print a formatted analysis summary."""
        print(f"\n{'='*60}")
        print(f"ANALYSIS: Knowledge of '{source_prompt}'")
        print(f"{'='*60}")
        
        summary = analysis["summary"]
        print(f"Total experiments: {summary['total_experiments']}")
        print(f"Successful patches: {summary['successful_patches']}")
        print(f"Expected keywords: {', '.join(summary['expected_keywords'])}")
        
        # Strong matches
        strong = analysis["strong_matches"]
        print(f"\n🎯 STRONG MATCHES ({len(strong)} results):")
        for result in strong:
            self._print_result_line(result)
        
        # Partial matches  
        partial = analysis["partial_matches"]
        print(f"\n⚡ PARTIAL MATCHES ({len(partial)} results):")
        for result in partial:
            self._print_result_line(result)
        
        # Weak matches
        weak = analysis["weak_matches"]
        if weak:
            print(f"\n❓ WEAK/UNCLEAR MATCHES ({len(weak)} results):")
            for result in weak[:5]:  # Limit display
                self._print_result_line(result)
            if len(weak) > 5:
                print(f"   ... and {len(weak) - 5} more")
    
    def _print_result_line(self, result: Dict[str, Any]):
        """Print a single result line."""
        extract_layer = result['extract_layer']
        inject_layer = result['inject_layer']
        new_tokens = result['new_tokens']
        
        # Truncate output for display
        max_len = ANALYSIS_CONFIG["output_max_length"]
        display_text = new_tokens[:max_len] + "..." if len(new_tokens) > max_len else new_tokens
        
        print(f"  E{extract_layer:2d}→I{inject_layer:2d}: {display_text}")
    
    def find_knowledge_hotspots(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify layer combinations that consistently produce good results.
        
        Args:
            analysis: Analysis results from analyze_results
            
        Returns:
            Hotspot analysis
        """
        strong_results = analysis["strong_matches"]
        
        if not strong_results:
            return {"message": "No strong matches found"}
        
        # Analyze extraction layers
        extract_layers = [r['extract_layer'] for r in strong_results]
        inject_layers = [r['inject_layer'] for r in strong_results]
        
        # Count occurrences
        extract_counts = defaultdict(int)
        inject_counts = defaultdict(int)
        pair_counts = defaultdict(int)
        
        for result in strong_results:
            ext = result['extract_layer']
            inj = result['inject_layer']
            extract_counts[ext] += 1
            inject_counts[inj] += 1
            pair_counts[(ext, inj)] += 1
        
        # Find hotspots
        hotspots = {
            "best_extract_layers": sorted(extract_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "best_inject_layers": sorted(inject_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "best_layer_pairs": sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            "extract_range": (min(extract_layers), max(extract_layers)) if extract_layers else None,
            "inject_range": (min(inject_layers), max(inject_layers)) if inject_layers else None
        }
        
        return hotspots
    
    def print_hotspot_analysis(self, hotspots: Dict[str, Any]):
        """Print hotspot analysis results."""
        print(f"\n📍 KNOWLEDGE HOTSPOT ANALYSIS")
        print("=" * 40)
        
        if "message" in hotspots:
            print(hotspots["message"])
            return
        
        print("🔥 Best extraction layers:")
        for layer, count in hotspots["best_extract_layers"]:
            print(f"   Layer {layer:2d}: {count} strong matches")
        
        print("\n💉 Best injection layers:")
        for layer, count in hotspots["best_inject_layers"]:
            print(f"   Layer {layer:2d}: {count} strong matches")
        
        print("\n🎯 Top layer pairs:")
        for (ext, inj), count in hotspots["best_layer_pairs"]:
            print(f"   E{ext:2d}→I{inj:2d}: {count} strong matches")
        
        if hotspots["extract_range"]:
            ext_min, ext_max = hotspots["extract_range"]
            inj_min, inj_max = hotspots["inject_range"]
            print(f"\n📊 Layer ranges:")
            print(f"   Extraction: {ext_min} to {ext_max}")
            print(f"   Injection: {inj_min} to {inj_max}")
    
    def compare_prompts(self, results_by_prompt: Dict[str, List[Dict[str, Any]]]):
        """
        Compare results across different source prompts.
        
        Args:
            results_by_prompt: Dictionary mapping source prompts to their results
        """
        print(f"\n🔍 CROSS-PROMPT COMPARISON")
        print("=" * 50)
        
        for source_prompt, results in results_by_prompt.items():
            analysis = self.analyze_results(results, source_prompt)
            summary = analysis["summary"]
            
            success_rate = (summary["strong_match_count"] / 
                          max(summary["successful_patches"], 1) * 100)
            
            print(f"\n{source_prompt}:")
            print(f"  Strong matches: {summary['strong_match_count']}/{summary['successful_patches']} "
                  f"({success_rate:.1f}%)")
            
            # Show best layer pairs for this prompt
            hotspots = self.find_knowledge_hotspots(analysis)
            if "best_layer_pairs" in hotspots and hotspots["best_layer_pairs"]:
                best_pair = hotspots["best_layer_pairs"][0]
                (ext, inj), count = best_pair
                print(f"  Best pair: E{ext}→I{inj} ({count} matches)")
