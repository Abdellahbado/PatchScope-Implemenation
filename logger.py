"""
Logging utilities for PatchScope experiments.
Handles detailed logging, history tracking, and result persistence.
"""

import os
import json
import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from config import LOGGING_CONFIG


class ExperimentLogger:
    """Handles logging and history tracking for PatchScope experiments."""
    
    def __init__(self, experiment_name: str = "patchscope_experiment"):
        self.experiment_name = experiment_name
        self.start_time = datetime.datetime.now()
        
        # Create relative paths that work everywhere
        self.base_dir = Path(".")
        self.history_file = self.base_dir / LOGGING_CONFIG["history_file"]
        self.results_file = self.base_dir / LOGGING_CONFIG["detailed_results_file"]
        self.summary_file = self.base_dir / LOGGING_CONFIG["summary_file"]
        
        # Initialize session
        self.session_id = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.experiment_data = {
            "session_id": self.session_id,
            "experiment_name": experiment_name,
            "start_time": self.start_time.isoformat(),
            "environment": self._detect_environment(),
            "events": [],
            "results": [],
            "summary": {}
        }
        
        # Start logging
        self._log_event("EXPERIMENT_START", f"Starting {experiment_name}")
        
    def _detect_environment(self) -> str:
        """Detect the execution environment."""
        if "COLAB_GPU" in os.environ:
            return "Google Colab"
        elif "/kaggle/" in str(Path.cwd()):
            return "Kaggle"
        elif "CODESPACE_NAME" in os.environ:
            return "GitHub Codespaces"
        else:
            return "Local"
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """Log model information."""
        self._log_event("MODEL_LOADED", f"Model: {model_info}")
        self.experiment_data["model_info"] = model_info
        
        # Print with more detail
        print(f"\n📊 MODEL CONFIGURATION")
        print("=" * 30)
        print(f"  Environment: {self.experiment_data['environment']}")
        print(f"  Session ID: {self.session_id}")
        print(f"  Structure: {model_info.get('structure_type', 'Unknown')}")
        print(f"  Total layers: {model_info.get('total_layers', 'Unknown')}")
        print(f"  Device: {model_info.get('device', 'Unknown')}")
        print(f"  Data type: {model_info.get('dtype', 'Unknown')}")
        
    def log_experiment_start(self, experiment_type: str, source_prompt: str, 
                           target_prompt: str, **kwargs):
        """Log the start of an experiment phase."""
        event_data = {
            "experiment_type": experiment_type,
            "source_prompt": source_prompt,
            "target_prompt": target_prompt,
            **kwargs
        }
        
        self._log_event("EXPERIMENT_PHASE_START", f"{experiment_type} experiment started", event_data)
        
        # Enhanced printing
        print(f"\n🔬 EXPERIMENT: {experiment_type.upper()}")
        print("=" * 50)
        print(f"  Source prompt: '{source_prompt}'")
        print(f"  Target template: '{target_prompt[:50]}...' (truncated)")
        if kwargs:
            for key, value in kwargs.items():
                if isinstance(value, list) and len(value) > 5:
                    print(f"  {key}: {value[:5]}... ({len(value)} total)")
                else:
                    print(f"  {key}: {value}")
    
    def log_single_patch_attempt(self, extract_layer: int, inject_layer: int, 
                                source_prompt: str, target_prompt: str):
        """Log a single patch attempt."""
        print(f"🔍 Testing E{extract_layer:2d}→I{inject_layer:2d}...", end=" ", flush=True)
        
    def log_single_patch_result(self, result: Optional[Dict[str, Any]], 
                              extract_layer: int, inject_layer: int):
        """Log the result of a single patch attempt."""
        if result and result.get('patch_applied', False):
            new_tokens = result['new_tokens']
            print(f"✅ Generated: '{new_tokens[:40]}{'...' if len(new_tokens) > 40 else ''}'")
            
            # Log detailed result
            self._log_event("PATCH_SUCCESS", 
                          f"E{extract_layer}→I{inject_layer}: {new_tokens}", 
                          result)
            self.experiment_data["results"].append(result)
        else:
            print("❌ Failed")
            self._log_event("PATCH_FAILED", f"E{extract_layer}→I{inject_layer}: Failed")
    
    def log_analysis_results(self, analysis: Dict[str, Any], source_prompt: str):
        """Log analysis results with enhanced detail."""
        summary = analysis["summary"]
        
        print(f"\n📈 ANALYSIS RESULTS")
        print("=" * 40)
        print(f"  Total experiments: {summary['total_experiments']}")
        print(f"  Successful patches: {summary['successful_patches']}")
        print(f"  Success rate: {summary['successful_patches']/max(summary['total_experiments'], 1)*100:.1f}%")
        print(f"  Strong matches: {summary['strong_match_count']}")
        print(f"  Partial matches: {summary['partial_match_count']}")
        print(f"  Weak matches: {summary['weak_match_count']}")
        
        # Log to file
        self._log_event("ANALYSIS_COMPLETE", f"Analysis for {source_prompt}", {
            "source_prompt": source_prompt,
            "summary": summary,
            "strong_matches_count": len(analysis["strong_matches"]),
            "best_results": analysis["strong_matches"][:3]  # Top 3
        })
        
        # Save detailed analysis
        self.experiment_data["analyses"] = self.experiment_data.get("analyses", [])
        self.experiment_data["analyses"].append({
            "source_prompt": source_prompt,
            "timestamp": datetime.datetime.now().isoformat(),
            "analysis": analysis
        })
    
    def log_hotspot_analysis(self, hotspots: Dict[str, Any]):
        """Log hotspot analysis results."""
        if "message" in hotspots:
            print(f"\n🔍 Hotspot Analysis: {hotspots['message']}")
            return
            
        print(f"\n🎯 KNOWLEDGE HOTSPOTS")
        print("=" * 30)
        
        # Best extraction layers
        if hotspots.get("best_extract_layers"):
            print("📤 Top extraction layers:")
            for layer, count in hotspots["best_extract_layers"][:3]:
                print(f"   Layer {layer:2d}: {count} strong matches")
        
        # Best injection layers  
        if hotspots.get("best_inject_layers"):
            print("📥 Top injection layers:")
            for layer, count in hotspots["best_inject_layers"][:3]:
                print(f"   Layer {layer:2d}: {count} strong matches")
        
        # Best pairs
        if hotspots.get("best_layer_pairs"):
            print("🎯 Best layer combinations:")
            for (ext, inj), count in hotspots["best_layer_pairs"][:5]:
                print(f"   E{ext:2d}→I{inj:2d}: {count} strong matches")
        
        # Log to file
        self._log_event("HOTSPOT_ANALYSIS", "Knowledge hotspots identified", hotspots)
    
    def log_experiment_summary(self, experiment_type: str, total_time: float, 
                             key_findings: Dict[str, Any]):
        """Log experiment summary and key findings."""
        print(f"\n🏁 EXPERIMENT SUMMARY: {experiment_type.upper()}")
        print("=" * 50)
        print(f"  Duration: {total_time:.1f} seconds")
        print(f"  Environment: {self.experiment_data['environment']}")
        
        if key_findings:
            print(f"  Key findings:")
            for key, value in key_findings.items():
                print(f"    {key}: {value}")
        
        # Update experiment data
        self.experiment_data["summary"] = {
            "experiment_type": experiment_type,
            "total_time": total_time,
            "key_findings": key_findings,
            "end_time": datetime.datetime.now().isoformat()
        }
        
        self._log_event("EXPERIMENT_COMPLETE", f"Completed {experiment_type}", key_findings)
    
    def save_session(self):
        """Save the complete session data."""
        try:
            # Save detailed JSON results
            with open(self.results_file, 'w') as f:
                json.dump(self.experiment_data, f, indent=2)
            
            # Create human-readable summary
            self._create_summary_file()
            
            print(f"\n💾 Session saved:")
            print(f"  History: {self.history_file}")
            print(f"  Detailed results: {self.results_file}")
            print(f"  Summary: {self.summary_file}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not save session: {e}")
    
    def _log_event(self, event_type: str, message: str, data: Dict[str, Any] = None):
        """Log an event to the history file."""
        timestamp = datetime.datetime.now().isoformat()
        
        event = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "event_type": event_type,
            "message": message,
            "data": data or {}
        }
        
        self.experiment_data["events"].append(event)
        
        # Append to history file for real-time tracking
        try:
            with open(self.history_file, 'a') as f:
                f.write(f"[{timestamp}] {event_type}: {message}\n")
                if data and LOGGING_CONFIG["enable_detailed_logging"]:
                    f.write(f"    Data: {json.dumps(data, indent=2)}\n")
        except Exception:
            pass  # Don't fail the experiment if logging fails
    
    def _create_summary_file(self):
        """Create a human-readable summary file."""
        with open(self.summary_file, 'w') as f:
            f.write(f"PatchScope Experiment Summary\n")
            f.write(f"=" * 40 + "\n\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Environment: {self.experiment_data['environment']}\n")
            f.write(f"Start Time: {self.start_time}\n")
            
            if "model_info" in self.experiment_data:
                model_info = self.experiment_data["model_info"]
                f.write(f"\nModel Information:\n")
                f.write(f"  Structure: {model_info.get('structure_type', 'Unknown')}\n")
                f.write(f"  Layers: {model_info.get('total_layers', 'Unknown')}\n")
                f.write(f"  Device: {model_info.get('device', 'Unknown')}\n")
            
            # Summary statistics
            total_results = len(self.experiment_data["results"])
            successful_patches = sum(1 for r in self.experiment_data["results"] 
                                   if r.get("patch_applied", False))
            
            f.write(f"\nExperiment Statistics:\n")
            f.write(f"  Total patch attempts: {total_results}\n")
            f.write(f"  Successful patches: {successful_patches}\n")
            if total_results > 0:
                f.write(f"  Success rate: {successful_patches/total_results*100:.1f}%\n")
            
            # Analysis summaries
            if "analyses" in self.experiment_data:
                f.write(f"\nAnalysis Results:\n")
                for analysis_data in self.experiment_data["analyses"]:
                    source = analysis_data["source_prompt"]
                    summary = analysis_data["analysis"]["summary"]
                    f.write(f"  {source}:\n")
                    f.write(f"    Strong matches: {summary['strong_match_count']}\n")
                    f.write(f"    Partial matches: {summary['partial_match_count']}\n")
            
            if "summary" in self.experiment_data:
                summary = self.experiment_data["summary"]
                f.write(f"\nKey Findings:\n")
                if "key_findings" in summary:
                    for key, value in summary["key_findings"].items():
                        f.write(f"  {key}: {value}\n")
