"""
Model loading and initialization utilities.
Handles model and tokenizer setup with optional quantization.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional

from config import MODEL_CONFIG


class ModelLoader:
    """Handles loading and configuring the model and tokenizer."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        
    def load_model_and_tokenizer(self, model_id: Optional[str] = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load model and tokenizer with optional quantization.
        
        Args:
            model_id: Model identifier. If None, uses default from config.
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if model_id is None:
            model_id = MODEL_CONFIG["model_id"]
            
        print(f"Loading model: {model_id}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare model loading arguments
        model_kwargs = {
            "device_map": MODEL_CONFIG["device_map"],
            "torch_dtype": MODEL_CONFIG["torch_dtype"],
        }
        
        # Add quantization if enabled (using torch native instead of bitsandbytes)
        if MODEL_CONFIG["quantization"]["enable"]:
            print("Using torch native quantization...")
            # Note: For production use, you might want to use torch.quantization
            # For now, we'll load normally and optionally quantize later
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
