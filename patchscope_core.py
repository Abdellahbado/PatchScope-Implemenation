"""
Core PatchScope implementation.
Handles the patching mechanism and single layer testing.
"""

import torch
from typing import Dict, Optional, Any
import time


class PatchScope:
    """Core PatchScope implementation for layer-wise intervention."""
    
    def __init__(self, model, tokenizer, model_loader):
        self.model = model
        self.tokenizer = tokenizer
        self.model_loader = model_loader
        
    def extract_representation(self, prompt: str, layer_idx: int) -> torch.Tensor:
        """
        Extract representation from a specific layer for the given prompt.
        
        Args:
            prompt: Source prompt to extract representation from
            layer_idx: Layer index to extract from (0-indexed)
            
        Returns:
            Extracted representation tensor
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Extract representation from the specified layer
        # Note: hidden_states[0] is embeddings, so layer_idx+1 for actual layers
        hidden_states = outputs.hidden_states[layer_idx + 1]
        representation = hidden_states[0, -1, :].clone().detach()
        
        # Also return the last token for debugging
        last_token = self.tokenizer.decode(inputs['input_ids'][0, -1])
        
        return representation, last_token
    
    def find_patch_position(self, target_prompt: str, marker: str = '?') -> Optional[int]:
        """
        Find the position of the patch marker in the target prompt.
        
        Args:
            target_prompt: Target prompt containing the marker
            marker: Marker to look for (default: '?')  # CHANGE THIS FROM 'x' TO '?'
            
        Returns:
            Position index or None if not found
        """
        inputs = self.tokenizer(target_prompt, return_tensors="pt").to(self.model.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Look for the marker token
        for i, token in enumerate(tokens):
            # Handle different tokenization formats
            if (token.lower().strip() == marker.lower() or 
                token in (marker, f' {marker}', f'Ġ{marker}', f'▁{marker}')):
                return i
                
        return None
    
    def run_single_patch(self, source_prompt: str, target_prompt: str, 
                        extract_layer: int, inject_layer: int) -> Optional[Dict[str, Any]]:
        """
        Run a single PatchScope experiment.
        """
        try:
            # Step 1: Extract source representation
            source_repr, source_token = self.extract_representation(source_prompt, extract_layer)
            
            # Debug print for extraction
            print(f"    📤 Extracted from layer {extract_layer}, last token: '{source_token}'")
            
            # Step 2: Find patch position
            patch_position = self.find_patch_position(target_prompt)  # This will now default to '?'
            if patch_position is None:
                print(f"    ❌ Could not find patch marker '?' in target prompt")  # UPDATE ERROR MESSAGE
                return None
                
            print(f"    📍 Patch position found at token index: {patch_position}")
            
            # Step 3: Prepare target inputs
            target_inputs = self.tokenizer(target_prompt, return_tensors="pt").to(self.model.device)
            
            # Step 4: Get target layer for injection
            layers = self.model_loader.get_layers()
            if inject_layer >= len(layers):
                print(f"    ❌ Inject layer {inject_layer} >= total layers {len(layers)}")
                return None
                
            target_layer = layers[inject_layer]
            print(f"    💉 Injecting into layer {inject_layer} ({type(target_layer).__name__})")
            
            # Step 5: Set up patching hook
            patch_applied = False
            
            def patching_hook(module, input_args, output):
                nonlocal patch_applied
                
                # Handle different output formats
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Apply patch if position is valid and not already applied
                if (hidden_states.shape[1] > patch_position and not patch_applied):
                    original_norm = torch.norm(hidden_states[0, patch_position, :]).item()
                    hidden_states[0, patch_position, :] = source_repr
                    new_norm = torch.norm(hidden_states[0, patch_position, :]).item()
                    print(f"    🔄 Patch applied! Norm change: {original_norm:.2f} → {new_norm:.2f}")
                    patch_applied = True
                
                # Return in original format
                if isinstance(output, tuple):
                    return (hidden_states,) + output[1:]
                else:
                    return hidden_states
            
            # Step 6: Apply hook and generate
            hook_handle = target_layer.register_forward_hook(patching_hook)
            
            try:
                print(f"    🎯 Generating with patch...")
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        target_inputs['input_ids'],
                        max_new_tokens=15,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Process results
                result_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # Extract new tokens
                original_length = len(target_prompt)
                if len(result_text) > original_length:
                    new_text = result_text[original_length:].strip()
                else:
                    new_text = ""
                
                print(f"    📝 Generated: '{new_text}' (patch_applied: {patch_applied})")
                
                return {
                    'extract_layer': extract_layer,
                    'inject_layer': inject_layer,
                    'patch_applied': patch_applied,
                    'generated_text': result_text,
                    'new_tokens': new_text,
                    'source_token': source_token,
                    'source_prompt': source_prompt,
                    'target_prompt': target_prompt,
                    'patch_position': patch_position,
                    'source_repr_norm': torch.norm(source_repr).item()
                }
                
            finally:
                hook_handle.remove()
                
        except Exception as e:
            print(f"    ❌ Error in patch (E{extract_layer}→I{inject_layer}): {e}")
            return None
    
    def run_layer_range_experiment(self, source_prompt: str, target_prompt: str,
                                 extract_layers: list, inject_layers: list) -> list:
        """
        Run PatchScope across a range of layer combinations.
        
        Args:
            source_prompt: Source prompt
            target_prompt: Target prompt  
            extract_layers: List of layers to extract from
            inject_layers: List of layers to inject into
            
        Returns:
            List of experiment results
        """
        results = []
        total_experiments = len(extract_layers) * len(inject_layers)
        
        print(f"Running {total_experiments} experiments...")
        print(f"Extract layers: {extract_layers}")
        print(f"Inject layers: {inject_layers}")
        
        experiment_count = 0
        
        for extract_layer in extract_layers:
            for inject_layer in inject_layers:
                experiment_count += 1
                
                result = self.run_single_patch(
                    source_prompt, target_prompt, extract_layer, inject_layer
                )
                
                if result:
                    results.append(result)
                    print(f"[{experiment_count:3d}/{total_experiments}] "
                          f"E{extract_layer:2d}→I{inject_layer:2d}: "
                          f"{result['new_tokens'][:30]}{'...' if len(result['new_tokens']) > 30 else ''}")
                else:
                    print(f"[{experiment_count:3d}/{total_experiments}] "
                          f"E{extract_layer:2d}→I{inject_layer:2d}: FAILED")
                
                # Small delay to avoid issues
                time.sleep(0.05)
        
        return results
