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
        "Mozart",
        "Napoleon Bonaparte",
        "Isaac Newton",
        "Charles Darwin",
        "Vincent van Gogh"
    ],
    
    # PatchScope templates with different styles and few-shot examples
    "patchscope_templates": [
        # Template 1: Original PatchScope style (mixed context)
        "Syria: Country in the Middle East. Leonardo DiCaprio: American actor. Samsung: South Korean multinational corporation. x:",
        
        # Template 2: Historical context style
        "Julius Caesar: Roman emperor who crossed the Rubicon. Napoleon Bonaparte: French emperor who was exiled to Elba. x:",
        
        # Template 3: Scientific context style  
        "Isaac Newton: Physicist who discovered gravity. Marie Curie: Chemist who discovered radium. x:",
        
        # Template 4: Geographic context style
        "Paris: Capital of France. Tokyo: Capital of Japan. x:",
        
        # Template 5: Artistic context style
        "Pablo Picasso: Spanish painter who created Guernica. Vincent van Gogh: Dutch painter who created Starry Night. x:",
    ],
    
    # Few-shot prompting templates with clear patterns
    "few_shot_templates": [
        # Template 1: Definition pattern (2-shot)
        "Albert Einstein: German physicist known for relativity theory. Charles Darwin: English naturalist known for evolution theory. x:",
        
        # Template 2: Achievement pattern (2-shot)
        "Mozart: Composed The Magic Flute and Don Giovanni. Beethoven: Composed the Ninth Symphony and Moonlight Sonata. x:",
        
        # Template 3: Historical role pattern (2-shot)
        "Cleopatra: Last pharaoh of ancient Egypt. Alexander the Great: Macedonian king who conquered the Persian Empire. x:",
        
        # Template 4: Invention/Discovery pattern (2-shot)
        "Thomas Edison: Invented the light bulb and phonograph. Alexander Graham Bell: Invented the telephone and audiometer. x:",
        
        # Template 5: Literary pattern (2-shot)
        "Shakespeare: Wrote Romeo and Juliet and Hamlet. Mark Twain: Wrote The Adventures of Tom Sawyer and Huckleberry Finn. x:",
        
        # Template 6: Scientific field pattern (3-shot)
        "Marie Curie: Pioneered radioactivity research. Isaac Newton: Developed laws of motion and gravity. Galileo: Advanced astronomy with telescopic observations. x:",
        
        # Template 7: Artistic style pattern (2-shot)
        "Pablo Picasso: Spanish painter who pioneered Cubism. Vincent van Gogh: Dutch post-impressionist known for bold colors. x:"
    ],
    
    # Context-free templates (minimal context for baseline comparison)
    "minimal_context_templates": [
        "The answer is x:",
        "x:",
        "Complete: x:",
        "Name: x:",
        "Entity: x:",
        "Describe x:",
        "What is x:"
    ],
    
    "target_templates": [
        # Template 1: Mixed context with clear marker (original PatchScope)
        "Syria: Country in the Middle East. Leonardo DiCaprio: American actor. Samsung: South Korean multinational corporation. x:",
        
        # Template 2: Simple context
        "The capital of France is Paris. The inventor of the telephone was Alexander Graham Bell. x:",
        
        # Template 3: Scientific context  
        "Water boils at 100°C. The speed of light is 299,792,458 m/s. x:",
        
        # Template 4: Historical context
        "World War II ended in 1945. The Roman Empire fell in 476 AD. x:",
        
        # Template 5: Geographic context
        "Mount Everest is the tallest mountain. The Pacific is the largest ocean. x:",
        
        # Template 6: Knowledge completion (as in paper)
        "Complete the following factual statement: x:",
        
        # Template 7: Entity description
        "Describe the following entity: x:",
        
        # Template 8: Direct completion
        "x:"
    ],
    
    # Contextual prompting templates (rich context for better results)
    "contextual_templates": [
        # Template 1: Professional context
        "Albert Einstein was a theoretical physicist. Marie Curie was a pioneering scientist. x:",
        
        # Template 2: Temporal context
        "The Renaissance occurred in the 14th-17th centuries. The Industrial Revolution occurred in the 18th-19th centuries. x:",
        
        # Template 3: Geographic context
        "Rome is the capital of Italy. Berlin is the capital of Germany. x:",
        
        # Template 4: Achievement context
        "Leonardo da Vinci painted the Mona Lisa. Michelangelo sculpted David. x:",
        
        # Template 5: Category context
        "Dogs are domesticated mammals. Cats are independent pets. x:"
    ],
    
    
    "analysis_keywords": {
        "George Washington": ["president", "first", "united states", "america", "washington", "founding father"],
        "Albert Einstein": ["physicist", "relativity", "scientist", "german", "theory"],
        "Marie Curie": ["physicist", "chemist", "nobel", "radioactivity", "scientist"],
        "Leonardo da Vinci": ["artist", "inventor", "renaissance", "mona lisa", "painter"],
        "Cleopatra": ["queen", "egypt", "pharaoh", "ancient", "egyptian"],
        "Aristotle": ["philosopher", "ancient", "greek", "logic", "ethics"],
        "Shakespeare": ["playwright", "english", "writer", "hamlet", "poet"],
        "Mozart": ["composer", "classical", "music", "austrian", "symphony"],
        "Napoleon Bonaparte": ["emperor", "french", "military", "conquest", "waterloo"],
        "Isaac Newton": ["physicist", "gravity", "laws", "mathematics", "scientist"],
        "Charles Darwin": ["evolution", "naturalist", "species", "origin", "scientist"],
        "Vincent van Gogh": ["painter", "dutch", "post-impressionist", "sunflowers", "artist"]
    },
    
    # Template categories for different experiment types
    "template_categories": {
        "patchscope": "patchscope_templates",
        "few_shot": "few_shot_templates", 
        "minimal": "minimal_context_templates",
        "contextual": "contextual_templates",
        "standard": "target_templates"
    }
}

# --- Logging Configuration ---
LOGGING_CONFIG = {
    "history_file": "experiment_history.log",  # Relative path for portability
    "detailed_results_file": "detailed_results.json",
    "summary_file": "experiment_summary.txt",
    "enable_detailed_logging": True,
    "log_level": "INFO"  # DEBUG, INFO, WARNING, ERROR
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
