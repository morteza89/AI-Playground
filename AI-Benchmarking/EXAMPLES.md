# Example Model Configurations

This file contains example configurations for popular models that work well with this benchmarking suite.

## Tested Model Configurations

### Small Models (Under 4GB)

**Microsoft Phi-3 Mini**
- HuggingFace ID: microsoft/Phi-3-mini-4k-instruct
- Recommended quantization: INT4
- Expected accuracy retention: 85-90%
- Typical speedup: 3-5x

Command:
```bash
optimum-cli export openvino --model microsoft/Phi-3-mini-4k-instruct --task text-generation-with-past --weight-format int4 --group-size 128 --ratio 1.0 --sym ov_phi3-mini_int4
```

### Medium Models (4-8GB)

**Qwen2-7B-Instruct**
- HuggingFace ID: Qwen/Qwen2-7B-Instruct
- Recommended quantization: INT4
- Expected accuracy retention: 90-95%
- Typical speedup: 3-4x

Command:
```bash
optimum-cli export openvino --model Qwen/Qwen2-7B-Instruct --task text-generation-with-past --weight-format int4 --group-size 128 --ratio 1.0 --sym ov_qwen2-7b_int4
```

**Llama 3.1 8B**
- HuggingFace ID: meta-llama/Meta-Llama-3.1-8B
- Recommended quantization: INT4
- Expected accuracy retention: 88-93%
- Typical speedup: 3-4x
- Note: May require HuggingFace authentication token

Command:
```bash
optimum-cli export openvino --model meta-llama/Meta-Llama-3.1-8B --task text-generation-with-past --weight-format int4 --group-size 128 --ratio 1.0 --sym ov_llama31-8b_int4
```

### Coding-Focused Models

**CodeLlama-7B-Instruct**
- HuggingFace ID: codellama/CodeLlama-7b-Instruct-hf
- Recommended quantization: INT4
- Expected accuracy retention: 85-92%
- Typical speedup: 3-4x
- Best for: MBPP coding benchmark

Command:
```bash
optimum-cli export openvino --model codellama/CodeLlama-7b-Instruct-hf --task text-generation-with-past --weight-format int4 --group-size 128 --ratio 1.0 --sym ov_codellama-7b_int4
```

### Large Models (Over 8GB)

**Gemma 2 9B IT**
- HuggingFace ID: google/gemma-2-9b-it
- Recommended quantization: INT4
- Expected accuracy retention: 90-95%
- Typical speedup: 2-3x
- Note: Requires authentication for download

Command:
```bash
optimum-cli export openvino --model google/gemma-2-9b-it --task text-generation-with-past --weight-format int4 --group-size 128 --ratio 1.0 --sym ov_gemma-2-9b_int4
```

## Quantization Options

### INT4 (Recommended for most cases)
- Weight format: int4
- Compression: 75% size reduction
- Accuracy loss: 5-10%
- Speed improvement: 3-5x
- Use case: Production deployment, resource-constrained environments

### INT8 (Better accuracy)
- Weight format: int8
- Compression: 50% size reduction
- Accuracy loss: 2-5%
- Speed improvement: 2-3x
- Use case: When accuracy is critical

### FP16 (Minimal loss)
- Weight format: fp16
- Compression: 50% size reduction
- Accuracy loss: 1-2%
- Speed improvement: 1.5-2x
- Use case: High-accuracy requirements, GPU deployment

## Advanced Quantization Parameters

### Group Size
Default: 128
- Smaller values (64): Better accuracy, slightly slower
- Larger values (256): Faster inference, slight accuracy loss

### Ratio
Default: 1.0
- Values less than 1.0: Mixed precision (some layers keep higher precision)
- Use 0.8 for critical layers in FP16 while others in INT4

### Symmetric Quantization
Flag: --sym
- Enables symmetric quantization
- Generally better for INT4
- Slightly faster inference

## Example Benchmark Configurations

### Quick Validation Test
- Sample count: 5 per dataset
- Hardware: CPU
- Model: Phi-3-mini (INT4)
- Time: 5-10 minutes

### Standard Evaluation
- Sample count: 10 per dataset
- Hardware: CPU or iGPU
- Model: Qwen2-7B (INT4)
- Time: 15-20 minutes

### Comprehensive Assessment
- Sample count: 20 per dataset
- Hardware: iGPU or NPU
- Model: Llama 3.1 8B (INT4)
- Benchmark: 5-dataset suite
- Time: 30-45 minutes

## Hardware Selection Guidelines

### CPU
Best for:
- Initial testing
- Consistent results
- Maximum compatibility
- Systems without dedicated AI hardware

### Intel iGPU
Best for:
- Laptops with integrated graphics
- Balanced performance
- Systems with Intel Core processors
- 2-3x speedup over CPU

### Intel NPU
Best for:
- Latest Intel Core Ultra processors
- Power efficiency
- Parallel AI workloads
- 3-5x speedup over CPU

## Benchmark Selection

### Use 3-Dataset Benchmark When:
- Quick validation needed
- Testing basic capabilities
- Comparing quantization methods
- Initial model evaluation

### Use 5-Dataset Benchmark When:
- Comprehensive evaluation required
- Testing coding capabilities
- Assessing truthfulness
- Production deployment decision
- Publishing benchmark results

## Expected Results by Model Size

### Small Models (Under 4B parameters)
- Accuracy: 70-85%
- Speedup: 4-6x
- Memory: 1-2GB
- TTFT: 100-300ms

### Medium Models (4-8B parameters)
- Accuracy: 80-92%
- Speedup: 3-5x
- Memory: 2-4GB
- TTFT: 200-500ms

### Large Models (Over 8B parameters)
- Accuracy: 85-95%
- Speedup: 2-4x
- Memory: 4-8GB
- TTFT: 300-800ms

## Common Quantization Commands Reference

Replace MODEL_NAME and OUTPUT_DIR with your values:

**INT4 Quantization:**
```bash
optimum-cli export openvino --model MODEL_NAME --task text-generation-with-past --weight-format int4 --group-size 128 --ratio 1.0 --sym OUTPUT_DIR
```

**INT8 Quantization:**
```bash
optimum-cli export openvino --model MODEL_NAME --task text-generation-with-past --weight-format int8 --sym OUTPUT_DIR
```

**FP16 Quantization:**
```bash
optimum-cli export openvino --model MODEL_NAME --task text-generation-with-past --weight-format fp16 OUTPUT_DIR
```

**Mixed Precision (INT4 with 80% ratio):**
```bash
optimum-cli export openvino --model MODEL_NAME --task text-generation-with-past --weight-format int4 --group-size 128 --ratio 0.8 --sym OUTPUT_DIR
```

## Notes

- Always test with your specific use case before production deployment
- Results may vary based on hardware and dataset
- Larger models generally have better accuracy retention after quantization
- INT4 offers the best balance of speed and accuracy for most applications
