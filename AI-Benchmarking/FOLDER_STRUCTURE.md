# AI-Benchmarking Folder Structure

This document describes the complete structure of the AI-Benchmarking folder ready for your GitHub repository.

## Folder Location

The AI-Benchmarking folder has been created at:
```
c:\Projects\optimizations\huggingf\AI-Benchmarking\
```

## Complete File List

1. **run_dynamic_dataset_test_general_model.py**
   - Main Python script for 3-dataset benchmark
   - Tests: MMLU, GSM8K, HellaSwag
   - Supports any HuggingFace model vs OpenVINO comparison

2. **run_dynamic_dataset_test_general_model.bat**
   - Windows batch file to launch 3-dataset benchmark
   - User-friendly interface with prompts

3. **run_dynamic_5benchmark_dataset_test_general_model.py**
   - Extended Python script for 5-dataset benchmark
   - Tests: MMLU, GSM8K, HellaSwag, MBPP (coding), TruthfulQA (honesty)
   - Comprehensive evaluation across all AI capability areas

4. **run_dynamic_5benchmark_dataset_test_general_model.bat**
   - Windows batch file to launch 5-dataset benchmark
   - Full capability assessment interface

5. **requirements.txt**
   - All required Python packages
   - Includes dependencies for model quantization (optimum-cli)
   - Compatible with pip install

6. **README.md**
   - Comprehensive documentation
   - Installation instructions
   - Usage guidelines
   - Troubleshooting section
   - No emojis or stickers, plain professional language

7. **QUICKSTART.md**
   - Step-by-step beginner guide
   - Complete walkthrough from installation to first benchmark
   - Example commands and expected results

8. **.gitignore**
   - Prevents committing large model files
   - Excludes cache and log files
   - Keeps repository clean

## Key Features

### Model Support
- Any HuggingFace text generation model
- OpenVINO quantized versions (INT4, INT8, FP16)
- Interactive model selection

### Hardware Support
- CPU (standard processing)
- Intel iGPU (integrated graphics)
- Intel NPU (neural processing unit)
- Automatic fallback to CPU if hardware unavailable

### Benchmark Capabilities
- Dynamic dataset loading from HuggingFace
- Random sampling for unbiased evaluation
- Flexible sample counts (5, 10, 20, or custom)
- Fresh random selection each run

### Metrics Provided
- Accuracy comparison (HuggingFace vs OpenVINO)
- TTFT (Time to First Token)
- Throughput (tokens per second)
- Token latency (milliseconds per token)
- Performance speedup calculations
- Production readiness assessment

### Documentation Quality
- Professional language throughout
- No emojis or decorative elements
- Clear step-by-step instructions
- Comprehensive troubleshooting
- Real-world examples

## How to Add to GitHub

1. Navigate to your AI-Playground repository
2. Create a new folder called "AI-Benchmarking"
3. Upload all 8 files from the local AI-Benchmarking directory
4. Commit with message: "Add AI model benchmarking suite"

## Folder Structure in Repository

Your repository will have:
```
AI-Playground/
├── Ollama-play-ground/
│   └── (existing files)
└── AI-Benchmarking/
    ├── .gitignore
    ├── QUICKSTART.md
    ├── README.md
    ├── requirements.txt
    ├── run_dynamic_dataset_test_general_model.bat
    ├── run_dynamic_dataset_test_general_model.py
    ├── run_dynamic_5benchmark_dataset_test_general_model.bat
    └── run_dynamic_5benchmark_dataset_test_general_model.py
```

## Requirements for Model Quantization

The requirements.txt includes all necessary packages for:

1. **Model Loading**
   - transformers (HuggingFace models)
   - torch (PyTorch backend)

2. **Dataset Access**
   - datasets (HuggingFace datasets)

3. **OpenVINO Integration**
   - openvino (inference engine)
   - optimum[openvino] (HuggingFace-OpenVINO bridge)
   - optimum-intel (Intel optimizations)

4. **Model Quantization** (optimum-cli)
   - nncf (neural network compression)
   - onnx (model format conversion)
   - onnxruntime (runtime support)

## Usage Example

After uploading to GitHub, users can:

```bash
# Clone the repository
git clone https://github.com/morteza89/AI-Playground.git

# Navigate to benchmarking suite
cd AI-Playground/AI-Benchmarking

# Install dependencies
pip install -r requirements.txt

# Quantize a model
optimum-cli export openvino --model Qwen/Qwen2-7B-Instruct --task text-generation-with-past --weight-format int4 --group-size 128 --ratio 1.0 --sym ov_qwen2-7b_int4

# Run benchmark
run_dynamic_5benchmark_dataset_test_general_model.bat
```

## Next Steps

1. Review all files in the AI-Benchmarking folder
2. Test locally to ensure everything works
3. Upload to your GitHub repository
4. Update repository README to link to AI-Benchmarking
5. Share with community

## Notes

- All documentation uses professional, human-readable language
- No emojis or decorative elements were used
- All instructions are clear and actionable
- Complete requirements for model quantization are included
- Files are ready for immediate GitHub upload
