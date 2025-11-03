# Quick Start Guide

This guide will help you run your first benchmark in under 10 minutes.

## Prerequisites Check

Before starting, verify you have:
- Python 3.8 or higher installed
- At least 10GB free disk space
- Internet connection for downloading models and datasets

## Step-by-Step Instructions

### 1. Install Required Packages

Open a terminal in the AI-Benchmarking directory and run:

```bash
pip install -r requirements.txt
```

Wait for all packages to install. This may take 5-10 minutes depending on your internet speed.

### 2. Prepare a Quantized Model

For this quick start, we will use Qwen2-7B-Instruct as an example.

Create a quantized INT4 version:

```bash
optimum-cli export openvino --model Qwen/Qwen2-7B-Instruct --task text-generation-with-past --weight-format int4 --group-size 128 --ratio 1.0 --sym ov_qwen2-7b_int4
```

This will download the model and create a quantized version in the "ov_qwen2-7b_int4" directory. This step may take 10-20 minutes.

### 3. Run Your First Benchmark

Start the basic 3-dataset benchmark:

```bash
run_dynamic_dataset_test_general_model.bat
```

Or on Linux/Mac:

```bash
python run_dynamic_dataset_test_general_model.py
```

### 4. Configure the Test

When prompted, enter:

**HuggingFace Model Selection:**
```
Qwen/Qwen2-7B-Instruct
```

**OpenVINO Model Path:**
```
./ov_qwen2-7b_int4
```

**Hardware Selection:**
Press Enter for CPU (most compatible)

**Sample Count:**
Press Enter for Standard test (10 samples per dataset)

### 5. Wait for Results

The benchmark will:
- Load both models
- Download test datasets
- Run accuracy evaluation
- Measure performance metrics

Total time: 10-15 minutes for standard test

### 6. Review Results

Results will be displayed on screen and saved to:
```
general_dynamic_dataset_test_results.log
```

Look for:
- Overall accuracy comparison
- Performance improvements
- Production readiness assessment

## What to Expect

### Typical Results for INT4 Quantization

Accuracy Retention: 85-95%
Performance Speedup: 2-4x faster
Memory Reduction: 50-75%

### Example Output

```
OVERALL ACCURACY (30 random samples total):
  HuggingFace: 27/30 (90.0%)
  OpenVINO:    25/30 (83.3%)

PERFORMANCE IMPROVEMENTS:
  TTFT: 3.2x faster
  Throughput: 4.1x faster
  Token Latency: 3.8x faster

PRODUCTION ASSESSMENT: PRODUCTION READY
```

## Next Steps

### Try the Comprehensive 5-Dataset Benchmark

For more thorough evaluation including coding and truthfulness:

```bash
run_dynamic_5benchmark_dataset_test_general_model.bat
```

### Test Different Quantization Methods

Compare INT4 vs INT8 vs FP16:

```bash
# INT8 (better accuracy, slower)
optimum-cli export openvino --model Qwen/Qwen2-7B-Instruct --weight-format int8 --sym ov_qwen2-7b_int8

# FP16 (minimal loss, less compression)
optimum-cli export openvino --model Qwen/Qwen2-7B-Instruct --weight-format fp16 --sym ov_qwen2-7b_fp16
```

Then benchmark each version to find the best balance.

### Try Different Models

Popular models to test:
- microsoft/Phi-3-mini-4k-instruct (smaller, faster)
- meta-llama/Meta-Llama-3.1-8B (balanced)
- codellama/CodeLlama-7b-Instruct-hf (coding-focused)

## Common Issues

### "Module not found" errors
Solution: Reinstall requirements
```bash
pip install --upgrade -r requirements.txt
```

### Model download is slow
Solution: Models are large (several GB). Be patient or use a faster internet connection.

### Out of memory errors
Solution: Try a smaller model or close other applications.

### Hardware selection fails
Solution: The benchmark will automatically fall back to CPU mode.

## Getting Help

If you encounter issues:
1. Check the detailed README.md file
2. Review the log files for error details
3. Ensure all prerequisites are met
4. Try with CPU mode first before using iGPU/NPU

## Summary

You have now:
- Installed all dependencies
- Created a quantized model
- Run your first benchmark
- Understood the results

You are ready to benchmark any HuggingFace model against its OpenVINO quantized version.
