@echo off
echo =============================================
echo OpenVINO GenAI 5-Benchmark Dataset Test
echo =============================================
echo This will benchmark your OpenVINO model
echo using openvino_genai optimized pipeline
echo on 5 comprehensive datasets with random sampling
echo.
echo Datasets: MMLU, GSM8K, HellaSwag, MBPP, TruthfulQA
echo Library: openvino_genai (faster inference)
echo.
python run_dynamic_5benchmark_dataset_test_general_OVmodel_GenAI.py
pause
