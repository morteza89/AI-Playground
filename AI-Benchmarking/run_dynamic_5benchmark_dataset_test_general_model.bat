@echo off
echo ================================================================================
echo GENERAL 5-BENCHMARK DYNAMIC DATASET MODEL TESTER
echo ================================================================================
echo Compare ANY HuggingFace model vs OpenVINO quantized version
echo Real dataset loading with random sampling each run
echo.
echo 5 COMPREHENSIVE BENCHMARKS:
echo   • MMLU (Academic Knowledge)
echo   • GSM8K (Mathematical Reasoning)
echo   • HellaSwag (Common Sense Reasoning)
echo   • MBPP (Code Generation)
echo   • TruthfulQA (Truthfulness ^& Honesty)
echo.
echo Requirements:
echo   - pip install datasets transformers optimum[openvino] torch
echo.
echo Usage Examples:
echo   - Qwen/Qwen2-7B-Instruct vs ./ov_qwen2-7b_int8
echo   - microsoft/Phi-3-mini-4k-instruct vs ./ov_phi3-mini_int4
echo   - google/gemma-2-2b-it vs ./ov_gemma-2-2b_int8
echo   - codellama/CodeLlama-7b-Instruct-hf vs ./ov_codellama-7b_int8
echo.
echo Comprehensive capability evaluation across all major AI areas!
echo Press any key to start...
pause > nul

python run_dynamic_5benchmark_dataset_test_general_model.py

echo.
echo ================================================================================
echo 5-Benchmark test completed!
echo Check general_5benchmark_dataset_test_results.log for full results
echo Comprehensive evaluation across Academic, Math, Common Sense, Coding, Honesty!
echo ================================================================================
pause