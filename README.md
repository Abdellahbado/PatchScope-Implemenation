# PatchScope Implementation

A research tool for analyzing knowledge representation in large language models. This implementation provides a systematic way to study how neural networks store and retrieve factual information through activation patching techniques.

## Overview

PatchScope is a method for understanding where and how language models process knowledge. It works by extracting neural activations from one context and injecting them into another context to observe the model's behavior.

### Core Methodology

1. **Extract** neural representations from a source context (e.g., "Einstein")
2. **Inject** these representations into a target context (e.g., "Describe the following person: ?")
3. **Analyze** the model's output to understand knowledge localization

## Installation

```bash
pip install torch transformers accelerate
```

## Quick Start

```bash
# Basic functionality test
python main.py --mode quick

# Specify a model
python main.py --model llama-3.2-3b-inst --mode quick

# Run comprehensive analysis
python main.py --model llama-3.2-3b-inst --mode comprehensive
```

## Supported Models

The implementation supports multiple model architectures:

| Model | Parameters | Memory Usage | Notes |
|-------|------------|--------------|-------|
| llama-3.2-1b-inst | 1B | ~3GB | Suitable for testing |
| llama-3.2-3b-inst | 3B | ~8GB | Recommended for research |
| llama-3.1-8b-inst | 8B | ~20GB | Best performance |
| phi-3-mini | 3.8B | ~8GB | Alternative architecture |

List all available models:
```bash
python list_models.py
```

## Experiment Types

### Quick Test
```bash
python main.py --mode quick
```
Validates basic functionality with a small set of layer combinations.

### Targeted Analysis
```bash
python main.py --mode targeted --source-prompt "Einstein"
```
Focuses on specific entities using predefined layer combinations.

### Comprehensive Study
```bash
python main.py --mode comprehensive
```
Runs multiple analysis phases including multi-prompt comparisons.

### Multi-Template Comparison
```bash
python main.py --mode templates --source-prompt "Einstein"
```
Tests different prompt templates to assess robustness.

## Key Research Findings

### Model Type Performance
- **Instruction-tuned models** consistently outperform base models
- **Base models** often fail to follow the experimental format
- **Recommendation**: Use instruction-tuned variants for reliable results

### Layer Injection Patterns
- **Early layers (10-30% of total)** show stronger patching effects
- **Late layers (70%+)** demonstrate minimal response to injection
- **Explanation**: Late layers have completed attention computations, limiting new information integration

### Optimal Configuration
```
Extract: Middle layers (40-60% of total layers)
Inject: Early layers (10-30% of total layers)
```

## Project Structure

```
├── main.py              # Command-line interface
├── config.py            # Model and experiment configuration
├── model_loader.py      # Model initialization utilities
├── patchscope_core.py   # Core patching algorithms
├── experiment_runner.py # Experiment orchestration
├── analysis.py          # Result analysis and visualization
├── logger.py            # Experiment logging system
├── experiment_history   # results from the experiment
└── README.md            # Documentation
```

## Configuration

### Model Configuration
Models are defined in `config.py` with specifications for:
- Model identifiers
- Memory allocation strategies
- Quantization settings
- Device mapping

### Experiment Parameters
Key parameters include:
- Layer selection ranges
- Token generation limits
- Analysis thresholds
- Logging verbosity

## Usage Examples

### Basic Entity Analysis
```bash
python main.py --model llama-3.2-3b-inst --mode targeted --source-prompt "Marie Curie"
```

### Multi-Entity Comparison
```bash
python main.py --model llama-3.2-3b-inst --mode multi
```

### Custom Template Testing
```bash
python main.py --mode patchscope --source-prompt "Newton" --template-type few_shot
```

## Output Interpretation

### Result Categories

**Strong Matches**: Generated text contains expected factual content
- Indicates successful knowledge retrieval
- Primary measure of patching effectiveness

**Partial Matches**: Generated text shows related but incomplete information
- Suggests partial knowledge activation
- Useful for understanding representation boundaries

**Failed Patches**: Generated text lacks relevant information
- May indicate inappropriate layer selection
- Suggests knowledge localization elsewhere

### Analysis Metrics
- Success rate across layer combinations
- Knowledge hotspot identification
- Cross-prompt consistency measures

## Technical Considerations

### Memory Requirements
- 1B models: 3-4GB GPU memory
- 3B models: 8-12GB GPU memory  
- 8B models: 20-24GB GPU memory

### Performance Optimization
- Automatic quantization for memory efficiency
- Batch processing for multiple experiments
- Cached model loading for repeated runs

## Troubleshooting

### Common Issues

**Marker Detection Failures**
- Ensure prompts contain the '?' placeholder
- Verify tokenizer compatibility with prompt format

**Out of Memory Errors**
- Reduce model size or enable quantization
- Limit concurrent experiments

**Poor Patching Results**
- Try different layer combinations
- Verify model type (instruction vs base)
- Check prompt template compatibility

## Research Applications

This implementation facilitates research in:
- Knowledge representation in neural networks
- Factual information localization
- Model interpretability and analysis
- Activation patching methodologies

## Contributing

Contributions are welcome in the following areas:
- Additional model architectures
- Enhanced analysis techniques
- Improved experiment designs
- Performance optimizations

## References

Based on the PatchScope methodology described in:
- Ghandeharioun, A., et al. "PatchScope: A Unifying Framework for Inspecting
Hidden Representations of Language Models"

## License

This implementation is provided for research purposes. Please cite appropriately when using in academic work.