@echo off
echo =============================================
echo OpenVINO 5-Benchmark Dataset Test
echo =============================================
echo This will benchmark your OpenVINO quantized model
echo on 5 comprehensive datasets with random sampling
echo.
echo Datasets: MMLU, GSM8K, HellaSwag, MBPP, TruthfulQA
echo.
python run_dynamic_5benchmark_dataset_test_general_OVmodel.py
pause
