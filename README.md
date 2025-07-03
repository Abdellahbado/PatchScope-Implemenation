# PatchScope Implementation - Enhanced Version

This project implements PatchScope as described in the paper, with a modular structure for easy experimentation and analysis, plus extensive logging and multiple experiment types.

## Structure

- `config.py` - Configuration settings, prompts, and parameters
- `model_loader.py` - Model and tokenizer loading utilities  
- `patchscope_core.py` - Core PatchScope implementation
- `analysis.py` - Result analysis and visualization
- `experiment_runner.py` - Experiment orchestration
- `logger.py` - **NEW**: Comprehensive logging system
- `main.py` - Main execution script with enhanced CLI
- `examples.py` - **NEW**: Usage examples and demonstrations

## Key Features

### **Enhanced PatchScope Experiments**
- **Multiple Template Types**: Test PatchScope with different prompt styles
- **Few-Shot Prompting**: Leverage pattern recognition with example-based templates
- **Template Comparison**: Compare performance across template types


### **Comprehensive Logging System**
All experiments automatically create detailed logs:
- `experiment_history.log` - Real-time event log with timestamps
- `detailed_results.json` - Complete structured results data  
- `experiment_summary.txt` - Human-readable summary

**Portable logging**: Uses relative paths, works in local/Colab/Kaggle environments.

### **Multiple Template Types for PatchScope**
```python
# PatchScope Templates (mixed context)
"Syria: Country in the Middle East. Leonardo DiCaprio: American actor. x:"

# Few-Shot Templates (pattern learning)
"Mozart: Composed The Magic Flute and Don Giovanni. Beethoven: Composed the Ninth Symphony and Moonlight Sonata. x:"

# Contextual Templates (rich context)
"Albert Einstein was a theoretical physicist. Marie Curie was a pioneering scientist. x:"

# Minimal Templates (baseline)
"The answer is x:"
```

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

### PatchScope with Template Types
```bash
python main.py --mode patchscope --source-prompt "Albert Einstein" --template-type few_shot
```

### Multi-Template Comparison
```bash
python main.py --mode templates --source-prompt "Marie Curie"
```

## Configuration

### Adding New Template Types
Edit `config.py` to add new template categories:

```python
PROMPTS = {
    "source_prompts": [
        "Your New Entity",  # Add here
        # ... existing prompts
    ],
    "patchscope_templates": [
        "Your new PatchScope template with x: marker",  # Add here
        # ... existing templates  
    ],
    "few_shot_templates": [
        "Example1: Description. Example2: Description. x:",  # Add here
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
