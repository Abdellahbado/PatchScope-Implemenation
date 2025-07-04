#!/usr/bin/env python3
"""
Utility script to list available models and their configurations.
"""

from config import MODEL_REGISTRY

def list_models():
    """List all available models."""
    print("🤖 AVAILABLE MODELS")
    print("=" * 50)
    
    for name, config in MODEL_REGISTRY.items():
        print(f"\n{name}:")
        print(f"  Model ID: {config['model_id']}")
        print(f"  Data Type: {config['torch_dtype']}")
        print(f"  Quantization: {config['quantization']['enable']}")
        
        # Estimate requirements
        if "1B" in config['model_id']:
            print(f"  Memory: ~2-3GB")
        elif "3B" in config['model_id']:
            print(f"  Memory: ~6-8GB")
        elif "8B" in config['model_id']:
            print(f"  Memory: ~16-20GB")
        else:
            print(f"  Memory: Variable")

if __name__ == "__main__":
    list_models()