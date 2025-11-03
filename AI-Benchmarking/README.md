# AI Model Benchmarking Suite

A comprehensive benchmarking toolkit for comparing HuggingFace language models against their OpenVINO quantized versions. This suite evaluates model performance across multiple dimensions including accuracy retention, inference speed, and hardware efficiency.

## Overview

This benchmarking suite provides two testing modes:

1. **3-Dataset Basic Benchmark** - Tests models on MMLU, GSM8K, and HellaSwag datasets
2. **5-Dataset Comprehensive Benchmark** - Extended testing including MBPP (coding) and TruthfulQA (honesty)

Both modes support:
- Dynamic dataset loading with random sampling
- Hardware selection (CPU, Intel iGPU, Intel NPU)
- Performance metrics (TTFT, throughput, token latency)
- Detailed accuracy comparison
- Production readiness assessment

## Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster HuggingFace model inference)
- Intel hardware (CPU, iGPU, or NPU) for OpenVINO inference

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Your Models

Before running benchmarks, you need both the original HuggingFace model and a quantized OpenVINO version.

#### Option A: Use Pre-trained HuggingFace Model

The benchmark will automatically download the HuggingFace model when you specify its ID (e.g., "meta-llama/Meta-Llama-3.1-8B").

#### Option B: Quantize Your Own Model

Use the OpenVINO optimization toolkit to create a quantized version:

```bash
optimum-cli export openvino --model MODEL_NAME --task text-generation-with-past --weight-format int4 --group-size 128 --ratio 1.0 --sym OUTPUT_DIR
```

Example for Qwen2-7B-Instruct with INT4 quantization:

```bash
optimum-cli export openvino --model Qwen/Qwen2-7B-Instruct --task text-generation-with-past --weight-format int4 --group-size 128 --ratio 1.0 --sym INT4-Qwen2-7B-instruct
```

Available weight formats:
- `int4` - 4-bit integer quantization (best compression)
- `int8` - 8-bit integer quantization (balanced)
- `fp16` - 16-bit floating point (minimal loss)

## Usage

### Basic 3-Dataset Benchmark

Tests model accuracy and performance on three fundamental capability areas:

```bash
run_dynamic_dataset_test_general_model.bat
```

Or run directly with Python:

```bash
python run_dynamic_dataset_test_general_model.py
```

**Datasets evaluated:**
- MMLU Mathematics - Academic knowledge
- GSM8K - Mathematical reasoning
- HellaSwag - Common sense reasoning

### Comprehensive 5-Dataset Benchmark

Extended evaluation including coding and truthfulness:

```bash
run_dynamic_5benchmark_dataset_test_general_model.bat
```

Or run directly with Python:

```bash
python run_dynamic_5benchmark_dataset_test_general_model.py
```

**Datasets evaluated:**
- MMLU Mathematics - Academic knowledge
- GSM8K - Mathematical reasoning
- HellaSwag - Common sense reasoning
- MBPP - Code generation
- TruthfulQA - Truthfulness and honesty

### Interactive Configuration

When you run either benchmark, you will be prompted to configure:

1. **HuggingFace Model Selection**
   - Enter the model ID (e.g., "Qwen/Qwen2-7B-Instruct")
   - Examples: meta-llama/Meta-Llama-3.1-8B, microsoft/Phi-3-mini-4k-instruct

2. **OpenVINO Model Path**
   - Provide the directory path to your quantized model
   - Example: ./INT4-Qwen2-7B-instruct

3. **Hardware Selection**
   - CPU - Standard CPU processing (most compatible)
   - iGPU - Intel Integrated Graphics (faster)
   - NPU - Intel Neural Processing Unit (most efficient)

4. **Sample Count**
   - Quick Test - 5 samples per dataset
   - Standard Test - 10 samples per dataset
   - Extended Test - 20 samples per dataset
   - Custom - Specify your own count

## Understanding Results

### Accuracy Metrics

The benchmark reports accuracy as a percentage of correctly answered questions:

- **HuggingFace Accuracy** - Baseline performance of the original model
- **OpenVINO Accuracy** - Performance of the quantized model
- **Accuracy Delta** - Difference between the two (positive means OpenVINO performed better)

### Performance Metrics

- **TTFT (Time to First Token)** - Latency before first token generation begins
- **Throughput** - Tokens generated per second
- **Token Latency** - Average time to generate each token
- **Speedup** - Performance improvement factor (e.g., 2.5x faster)

### Production Assessment

The benchmark provides an automated assessment:

- **Production Ready** - Accuracy above 70% with performance improvements
- **Requires Review** - Accuracy between 50-70%, needs evaluation
- **Needs Improvement** - Accuracy below 50%, quantization may be too aggressive

## Output Files

Results are automatically saved to log files:

- `general_dynamic_dataset_test_results.log` - Basic 3-dataset results
- `general_5benchmark_dataset_test_results.log` - Comprehensive 5-dataset results

These files contain detailed information including:
- Individual sample results
- Response comparisons
- Performance measurements
- Hardware utilization details

## Example Workflow

### Testing a New Model

1. Choose your model from HuggingFace (e.g., Qwen/Qwen2-7B-Instruct)

2. Quantize it using optimum-cli:
```bash
optimum-cli export openvino --model Qwen/Qwen2-7B-Instruct --task text-generation-with-past --weight-format int4 --group-size 128 --ratio 1.0 --sym ./ov_qwen2-7b_int4
```

3. Run the comprehensive benchmark:
```bash
run_dynamic_5benchmark_dataset_test_general_model.bat
```

4. Enter configuration:
   - HuggingFace Model: Qwen/Qwen2-7B-Instruct
   - OpenVINO Model: ./ov_qwen2-7b_int4
   - Hardware: CPU (or iGPU/NPU if available)
   - Sample Count: Standard (10 samples per dataset)

5. Review results and decide on production readiness

### Comparing Quantization Methods

Test different quantization approaches:

```bash
# INT4 quantization
optimum-cli export openvino --model MODEL_NAME --weight-format int4 --sym ./model_int4

# INT8 quantization
optimum-cli export openvino --model MODEL_NAME --weight-format int8 --sym ./model_int8

# FP16 quantization
optimum-cli export openvino --model MODEL_NAME --weight-format fp16 --sym ./model_fp16
```

Then benchmark each version to compare accuracy retention vs performance gains.

## Hardware Recommendations

### CPU
- Best compatibility
- Reliable performance
- Recommended for initial testing

### Intel iGPU
- Faster inference
- Good for laptops with integrated graphics
- May require Intel GPU drivers

### Intel NPU
- Most power-efficient
- Specialized hardware acceleration
- Requires newer Intel processors with NPU support

## Troubleshooting

### Import Errors

If you encounter import errors, ensure all dependencies are installed:

```bash
pip install --upgrade -r requirements.txt
```

### Model Loading Failures

- Verify the HuggingFace model ID is correct
- Check that the OpenVINO model path exists
- Ensure sufficient disk space for model downloads

### Hardware Selection Issues

If your selected hardware fails:
- The benchmark will automatically fall back to CPU
- Check Intel driver updates for iGPU/NPU support
- Verify hardware compatibility with OpenVINO

### Performance Measurement Errors

If performance measurement fails:
- Results will use fallback values
- Check log files for detailed error messages
- Consider reducing sample count for initial tests

## Advanced Usage

### Custom Dataset Integration

To add your own datasets, modify the `dataset_configs` dictionary in the Python files:

```python
"YourDataset": {
    "name": "Your Dataset Name",
    "dataset_name": "huggingface/dataset-id",
    "subset": "subset_name",
    "description": "Description of dataset",
    "question_key": "question_field",
    "answer_key": "answer_field",
    "category": "Dataset Category"
}
```

### Automated Testing

For CI/CD integration, you can bypass interactive prompts by modifying the model selection methods or creating wrapper scripts with predefined configurations.

## Citation

If you use this benchmarking suite in your research, please cite:

```
AI Model Benchmarking Suite
https://github.com/morteza89/AI-Playground/tree/main/AI-Benchmarking
```

## License

This project is provided as-is for research and development purposes.

## Contributing

Contributions are welcome. Please ensure:
- Code follows existing style conventions
- New features include appropriate documentation
- Testing is performed on multiple hardware configurations

## Additional Documentation

For more detailed information, see the documentation in the `docs/` folder:

- **[Quick Start Guide](docs/QUICKSTART.md)** - Step-by-step beginner tutorial
- **[Model Examples](docs/EXAMPLES.md)** - Configuration examples for popular models
- **[Folder Structure](docs/FOLDER_STRUCTURE.md)** - Detailed project organization

Example outputs from real benchmark runs are available in the `outputs/` folder.

## Support

For issues or questions:
- Check the troubleshooting section above
- Review log files for detailed error information
- Consult the additional documentation in the `docs/` folder
- Open an issue on the GitHub repository
