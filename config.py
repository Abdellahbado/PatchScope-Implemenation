"""
Configuration module for PatchScope experiments.
Contains all configurable parameters and prompts.
"""

import torch

# --- Model Configuration ---
MODEL_CONFIG = {
    "model_id": "meta-llama/Llama-3.2-3B-Instruct",
    "device_map": "auto",
    "torch_dtype": torch.bfloat16,
    # Using torch native quantization instead of bitsandbytes
    "quantization": {
        "enable": True,
        "dtype": torch.int8  # Using torch.quantization instead of bitsandbytes
    }
}

# --- Experiment Configuration ---
EXPERIMENT_CONFIG = {
    "max_new_tokens": 15,
    "do_sample": False,
    "temperature": 0.0,
    "layer_step": 2,  # Test every 2nd layer initially
    "detailed_range": 3,  # How many layers around hotspots to test in detail
}

# --- Prompts Configuration ---
# Easy to edit prompts as guided by the paper
PROMPTS = {
    "source_prompts": [
        "George Washington",
        "Albert Einstein", 
        "Marie Curie",
        "Leonardo da Vinci",
        "Cleopatra",
        "Aristotle",
        "Shakespeare",
        "Mozart"
    ],
    
    "target_templates": [
        # Template 1: Mixed context with clear marker
        "Syria: Country in the Middle East. Leonardo DiCaprio: American actor. Samsung: South Korean multinational corporation. x:",
        
        # Template 2: Simple context
        "The capital of France is Paris. The inventor of the telephone was Alexander Graham Bell. x:",
        
        # Template 3: Scientific context  
        "Water boils at 100°C. The speed of light is 299,792,458 m/s. x:",
        
        # Template 4: Historical context
        "World War II ended in 1945. The Roman Empire fell in 476 AD. x:",
        
        # Template 5: Geographic context
        "Mount Everest is the tallest mountain. The Pacific is the largest ocean. x:"
    ],
    
    "analysis_keywords": {
        "George Washington": ["president", "first", "united states", "america", "washington", "founding father"],
        "Albert Einstein": ["physicist", "relativity", "scientist", "german", "theory"],
        "Marie Curie": ["physicist", "chemist", "nobel", "radioactivity", "scientist"],
        "Leonardo da Vinci": ["artist", "inventor", "renaissance", "mona lisa", "painter"],
        "Cleopatra": ["queen", "egypt", "pharaoh", "ancient", "egyptian"],
        "Aristotle": ["philosopher", "ancient", "greek", "logic", "ethics"],
        "Shakespeare": ["playwright", "english", "writer", "hamlet", "poet"],
        "Mozart": ["composer", "classical", "music", "austrian", "symphony"]
    }
}

# --- Layer Configuration ---
LAYER_CONFIG = {
    "total_layers": 28,  # Llama-3.2-3B has 28 layers
    "early_layers": list(range(0, 7)),      # 0-6: Basic representations
    "mid_layers": list(range(7, 21)),       # 7-20: Semantic processing  
    "late_layers": list(range(21, 28)),     # 21-27: High-level reasoning
    
    # Specific layer ranges for targeted testing
    "targeted_layers": [2, 7, 14, 21, 26],  # Representative layers from each section
}

# --- Analysis Configuration ---
ANALYSIS_CONFIG = {
    "match_thresholds": {
        "strong_match": 2,    # Keywords needed for strong match
        "partial_match": 1,   # Keywords needed for partial match
    },
    "output_max_length": 50,  # Max length for display in summaries
}
