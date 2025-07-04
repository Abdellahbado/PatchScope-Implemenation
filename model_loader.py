"""
Model loading and initialization utilities.
Handles model and tokenizer setup with optional quantization.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional

# Update the ModelLoader class in model_loader.py

from config import MODEL_CONFIG, get_model_config

class ModelLoader:
    """Handles loading and configuring the model and tokenizer."""
    
    def __init__(self, model_name: str = None):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_name = model_name
        
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
        # Priority: direct model_id > model_name > instance model_name > default
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
        """Get information about the loaded model structure."""
        if self.model is None:
            return {"error": "No model loaded"}
            
        try:
            # Determine model structure and layer count
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                layers = self.model.model.layers
                structure_type = "model.model.layers"
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                layers = self.model.transformer.h
                structure_type = "model.transformer.h"
            else:
                return {"error": "Unknown model structure"}
                
            return {
                "structure_type": structure_type,
                "total_layers": len(layers),
                "layer_type": type(layers[0]).__name__,
                "device": str(self.device),
                "dtype": str(self.model.dtype)
            }
            
        except Exception as e:
            return {"error": f"Error analyzing model: {e}"}
    
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
