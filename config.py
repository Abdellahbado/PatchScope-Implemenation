"""
Configuration module for PatchScope experiments.
Contains all configurable parameters and prompts.
"""

import torch

# --- Model Configuration ---
# Add this to config.py after the existing MODEL_CONFIG

# --- Model Registry ---
MODEL_REGISTRY = {
    "llama-3.2-3b-inst": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "quantization": {"enable": True, "dtype": torch.int8}
    },
    "llama-3.2-3b": {
        "model_id": "meta-llama/Llama-3.2-3B",
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "quantization": {"enable": True, "dtype": torch.int8}
    }
}

# Default model (keep existing behavior)
DEFAULT_MODEL = "llama-3.2-3b"

def get_model_config(model_name: str = None) -> dict:
    """Get model configuration by name."""
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    if model_name not in MODEL_REGISTRY:
        available_models = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
    
    return MODEL_REGISTRY[model_name]

# --- Experiment Configuration ---
EXPERIMENT_CONFIG = {
    "max_new_tokens": 15,
    "do_sample": False,
    "temperature": 0.0,
    "layer_step": 2,  # Test every 2nd layer initially
    "detailed_range": 3,  # How many layers around hotspots to test in detail
}

# --- Prompts Configuration ---
# Enhanced prompts based on best practices for clarity, specificity, and structure.
PROMPTS = {
    # Suggestions: Diversified the list to include figures from different fields
    # and with varying name structures. This helps in testing the model's
    # robustness in entity recognition.
    "source_prompts": [
        "George Washington",
        "Albert Einstein",
        "Marie Curie",
        "Leonardo da Vinci",
        "Cleopatra VII",  # Added specificity
        "Aristotle",
        "William Shakespeare",  # Used full name for consistency
        "Wolfgang Amadeus Mozart", # Used full name
        "Napoleon Bonaparte",
        "Isaac Newton",
        "Charles Darwin",
        "Vincent van Gogh",
        "Ada Lovelace", # Added a prominent female figure in tech
        "Martin Luther King Jr.", # Included a historical figure with a suffix
    ],

    # Suggestions: The original templates were good but mixed different contexts.
    # These have been refined for clarity and to provide a clearer signal to the model.
    # The use of "?:" is a good placeholder as noted in the paper.
    "patchscope_templates": [
        # Template 1: More explicit instruction for description
        "Provide a brief description of the following entity. Syria: A country in the Middle East. Leonardo DiCaprio: An American actor. Samsung: A South Korean multinational corporation. ?:",

        # Template 2: Clearer historical context with a direct command
        "Describe the historical significance of the person. Julius Caesar: A Roman general and statesman who played a critical role in the demise of the Roman Republic. Napoleon Bonaparte: A French military and political leader who rose to prominence during the French Revolution. ?:",

        # Template 3: Focused scientific contribution
        "Explain the main scientific contribution of the individual. Isaac Newton: Developed the laws of motion and universal gravitation. Marie Curie: Conducted pioneering research on radioactivity. ?:",

        # Template 4: Simple and direct entity identification
        "Identify the following entity. Paris: The capital city of France. Tokyo: The capital city of Japan. ?:",

        # Template 5: Artistic context with a focus on their work
        "Describe the artist and their notable work. Pablo Picasso: A Spanish painter who co-founded the Cubist movement and created 'Guernica'. Vincent van Gogh: A Dutch Post-Impressionist painter who created 'The Starry Night'. ?:",
    ],

    # Suggestions: The few-shot templates are already quite strong. The edits below
    # focus on making the pattern even more explicit and consistent.
    "few_shot_templates": [
        # Template 1: Clearer 'Person: Description' pattern
        "Albert Einstein: A German-born theoretical physicist who developed the theory of relativity. Charles Darwin: An English naturalist, geologist, and biologist, best known for his contributions to the science of evolution. ?:",

        # Template 2: Consistent 'Composer: Notable Works' pattern
        "Wolfgang Amadeus Mozart: Composer of 'The Magic Flute' and 'Don Giovanni'. Ludwig van Beethoven: Composer of the 'Ninth Symphony' and 'Moonlight Sonata'. ?:",

        # Template 3: Consistent 'Historical Figure: Role' pattern
        "Cleopatra: The last active ruler of the Ptolemaic Kingdom of Egypt. Alexander the Great: A king of the ancient Greek kingdom of Macedon. ?:",

        # Template 4: Consistent 'Inventor: Inventions' pattern
        "Thomas Edison: Inventor of the practical electric light bulb and the phonograph. Alexander Graham Bell: Inventor of the telephone. ?:",

        # Template 5: Consistent 'Author: Major Works' pattern
        "William Shakespeare: Author of 'Romeo and Juliet' and 'Hamlet'. Mark Twain: Author of 'The Adventures of Tom Sawyer' and 'Adventures of Huckleberry Finn'. ?:",
    ],

    # Suggestions: Minimal context prompts are for baseline testing and should be simple.
    # The original prompts are good. Added a couple more variations.
    "minimal_context_templates": [
        "The answer is ?:",
        "?:",
        "Complete: ?:",
        "Name: ?:",
        "Entity: ?:",
        "Describe ?:",
        "What is ?:",
        "Information about ?:", # Added variation
        "Here is information on ?:", # Added variation
    ],

    # Suggestions: Target templates from the paper are designed to extract specific information.
    # The revised templates are more direct and ask for a specific piece of information.
    "target_templates": [
        # Template 1: Direct instruction for entity description.
        "Describe the following entity in one sentence: ?:",

        # Template 2: More specific knowledge completion.
        "Complete the following fact. ?:",

        # Template 3: Clearer role-playing prompt.
        "You are a helpful assistant. Provide a summary of the entity ?:",

        # Template 4: Structured output request.
        "Provide the following information for ?: \n- Field: \n- Main Achievement:",

        # Template 5: Simple and direct completion.
        "? is known for",
    ],

    # Suggestions: Contextual templates should provide rich, relevant context to guide the model.
    # The revised templates offer more descriptive context.
    "contextual_templates": [
        # Template 1: Richer professional context.
        "In the context of influential scientists who revolutionized our understanding of the universe, consider the following. Albert Einstein was a theoretical physicist who developed the theory of relativity. Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity. Now, describe ?:",

        # Template 2: More descriptive temporal context.
        "During the Renaissance, a period of great cultural change and artistic development in Europe, Leonardo da Vinci was a painter, architect, and inventor. In the same era, Michelangelo was a sculptor, painter, and architect. In this context, who was ?:",

        # Template 3: More specific geographic context.
        "In the history of major European capitals, Rome is the capital of Italy and is known for its ancient Roman ruins. Berlin is the capital of Germany, known for its vibrant arts scene and historical significance. In this context, what is ?:",

        # Template 4: Detailed achievement context.
        "Within the realm of great artistic achievements, Leonardo da Vinci painted the 'Mona Lisa', one of the world's most famous portraits. Michelangelo sculpted 'David', a masterpiece of High Renaissance sculpture. Following this pattern of artists and their masterpieces, describe ?:",
    ],

    "analysis_keywords": {
        "George Washington": ["president", "first", "united states", "america", "washington", "founding father", "general"],
        "Albert Einstein": ["physicist", "relativity", "scientist", "german", "e=mc^2", "theory of relativity"],
        "Marie Curie": ["physicist", "chemist", "nobel prize", "radioactivity", "scientist", "polish", "french"],
        "Leonardo da Vinci": ["artist", "inventor", "renaissance", "mona lisa", "painter", "italian"],
        "Cleopatra VII": ["queen", "egypt", "pharaoh", "ptolemaic", "egyptian", "ruler"],
        "Aristotle": ["philosopher", "ancient", "greek", "logic", "ethics", "student of plato"],
        "William Shakespeare": ["playwright", "english", "writer", "hamlet", "poet", "bard"],
        "Wolfgang Amadeus Mozart": ["composer", "classical", "music", "austrian", "symphony", "opera"],
        "Napoleon Bonaparte": ["emperor", "french", "military", "leader", "waterloo"],
        "Isaac Newton": ["physicist", "gravity", "laws of motion", "mathematics", "scientist", "english"],
        "Charles Darwin": ["evolution", "naturalist", "on the origin of species", "biologist", "scientist"],
        "Vincent van Gogh": ["painter", "dutch", "post-impressionist", "starry night", "artist"],
        "Ada Lovelace": ["mathematician", "writer", "charles babbage", "analytical engine", "programmer"],
        "Martin Luther King Jr.": ["civil rights", "leader", "activist", "i have a dream", "minister"],
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