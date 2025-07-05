"""
Model loading and initialization utilities.
Handles model and tokenizer setup with optional quantization.
"""

"""
Model loading and initialization utilities.
Handles model and tokenizer setup with optional quantization.
"""

import torch
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

# Fix the import - remove MODEL_CONFIG, keep get_model_config
from config import get_model_config, MODEL_REGISTRY, DEFAULT_MODEL

class ModelLoader:
    """Handles loading and configuring the model and tokenizer."""
    
    def __init__(self, model_name: str = None):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_name = model_name or DEFAULT_MODEL
        
    def load_model_and_tokenizer(self, model_id: Optional[str] = None, 
                                model_name: Optional[str] = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load model and tokenizer with optional quantization.
        
        Args:
            model_id: Direct model identifier (takes precedence)
            model_name: Model name from registry (e.g., "llama-3.2-3b")
            
        Returns:
            Tuple of (model, tokenizer)
        """
        # Priority: direct model_id > model_name parameter > instance model_name > default
        if model_id:
            model_config = {
                "model_id": model_id,
                "device_map": "auto",
                "torch_dtype": torch.bfloat16,
                "quantization": {"enable": True, "dtype": torch.int8}
            }
        else:
            # Use model_name parameter or instance model_name
            selected_model_name = model_name or self.model_name
            model_config = get_model_config(selected_model_name)
        
        model_id = model_config["model_id"]
        print(f"Loading model: {model_id}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare model loading arguments
        model_kwargs = {
            "device_map": model_config["device_map"],
            "torch_dtype": model_config["torch_dtype"],
        }
        
        # Add quantization if enabled
        if model_config["quantization"]["enable"]:
            print("Using torch native quantization...")
            # Note: Actual quantization implementation would go here
            pass
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        
        # Store references
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        
        print("Model and tokenizer loaded successfully.")
        return model, tokenizer
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {"error": "No model loaded"}
        
        # Detect model structure
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            structure_type = "model.model.layers"
            total_layers = len(self.model.model.layers)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            structure_type = "transformer.h"
            total_layers = len(self.model.transformer.h)
        else:
            structure_type = "unknown"
            total_layers = 0
        
        return {
            "structure_type": structure_type,
            "total_layers": total_layers,
            "device": str(self.device),
            "model_name": self.model_name,
            "dtype": str(self.model.dtype) if hasattr(self.model, 'dtype') else "unknown"
        }   
    def get_layers(self):
        """Get the model layers for patching."""
        if self.model is None:
            raise ValueError("No model loaded")
            
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        else:
            raise ValueError("Could not access model layers")
