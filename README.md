# PatchScope Implementation

This project implements PatchScope as described in the paper, with a modular structure for easy experimentation and analysis.

## Structure

- `config.py` - Configuration settings, prompts, and parameters
- `model_loader.py` - Model and tokenizer loading utilities  
- `patchscope_core.py` - Core PatchScope implementation
- `analysis.py` - Result analysis and visualization
- `experiment_runner.py` - Experiment orchestration
- `main.py` - Main execution script

## Key Features

### Variable Layer Combinations
- Extract from layer X, inject to layer Y (not just X→X)
- Strategic layer combination testing to avoid quadratic complexity
- Focused on most informative layer pairs

### Easy-to-Edit Prompts
- Multiple source prompts for different entities
- Various target prompt templates
- Configurable analysis keywords for each entity

### No BitsAndBytes Dependency  
- Uses torch native quantization instead
- More compatible across different environments

### Modular Design
- Each component has a specific responsibility
- Easy to modify or extend individual parts
- Clear separation of concerns

## Usage

### Quick Test
```bash
python main.py --mode quick
```

### Targeted Study (specific layers)
```bash
python main.py --mode targeted --source-prompt "Albert Einstein"
```

### Sweep Study (strategic layer combinations)
```bash
python main.py --mode sweep --source-prompt "Marie Curie" 
```

### Comprehensive Study (all phases)
```bash
python main.py --mode comprehensive
```

### Multi-Prompt Comparison
```bash
python main.py --mode multi
```

## Configuration

### Adding New Prompts
Edit `config.py` to add new source prompts or target templates:

```python
PROMPTS = {
    "source_prompts": [
        "Your New Entity",  # Add here
        # ... existing prompts
    ],
    "target_templates": [
        "Your new template with x: marker",  # Add here
        # ... existing templates  
    ],
    "analysis_keywords": {
        "Your New Entity": ["keyword1", "keyword2", "keyword3"]
    }
}
```

### Adjusting Layer Testing
Modify `LAYER_CONFIG` in `config.py`:

```python
LAYER_CONFIG = {
    "targeted_layers": [2, 7, 14, 21, 26],  # Key layers to test
    # ... other configurations
}
```

### Experiment Parameters
Adjust `EXPERIMENT_CONFIG` in `config.py` for generation settings.

## Layer Combination Strategy

Instead of testing all 28×28=784 combinations, the implementation uses:

1. **Targeted Testing**: Strategic layer pairs from different regions
2. **Sweep Testing**: Diverse combinations covering the layer space
3. **Hotspot Analysis**: Focus on promising layer combinations
4. **Progressive Refinement**: Start broad, then zoom into interesting regions

## Analysis Features

- **Match Classification**: Strong/partial/weak matches based on keyword presence
- **Hotspot Detection**: Identifies best extraction and injection layers  
- **Cross-Prompt Comparison**: Compare knowledge patterns across entities
- **Statistical Summary**: Success rates and pattern analysis

## Dependencies

Install required packages:
```bash
pip install torch transformers accelerate
```

No `bitsandbytes` required - uses torch native quantization for memory efficiency.
