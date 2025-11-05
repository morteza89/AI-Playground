@echo off
REM HuggingFace vs OpenVINO GenAI 5-Benchmark Comparison Test
REM Compares HF models against OpenVINO GenAI library (optimized pipeline)
REM Tests: MMLU, GSM8K, HellaSwag, MBPP (Coding), TruthfulQA (Honesty)

echo ================================================================================
echo HUGGINGFACE vs OPENVINO GENAI 5-BENCHMARK COMPARISON
echo ================================================================================
echo.
echo This script compares HuggingFace models against OpenVINO GenAI pipeline
echo Comprehensive benchmarking across 5 capability areas
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Run the HF vs GenAI comparison test
echo Running HuggingFace vs OpenVINO GenAI 5-Benchmark Comparison...
echo.
python run_dynamic_5benchmark_dataset_test_general_modelvsGenAI.py

REM Check if the script executed successfully
if errorlevel 1 (
    echo.
    echo ================================================================================
    echo TEST FAILED
    echo ================================================================================
    echo Check the error messages above for details
    pause
    exit /b 1
) else (
    echo.
    echo ================================================================================
    echo TEST COMPLETED SUCCESSFULLY
    echo ================================================================================
    echo Results saved to: outputs\^<model_name^>\general_5benchmark_hf_vs_genai_test_results.log
    echo.
    pause
)
