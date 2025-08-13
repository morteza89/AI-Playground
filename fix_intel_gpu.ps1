# PowerShell script to fix Intel GPU support for Ollama
Write-Host "=== Ollama Intel GPU Fix Script ===" -ForegroundColor Green

# Step 1: Kill all Ollama processes
Write-Host "Step 1: Stopping all Ollama processes..." -ForegroundColor Yellow
Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 3

# Step 2: Check Intel GPU
Write-Host "Step 2: Checking Intel GPU..." -ForegroundColor Yellow
$intelGPU = Get-WmiObject Win32_VideoController | Where-Object {$_.Name -like "*Intel*"}
if ($intelGPU) {
    Write-Host "✓ Found Intel GPU: $($intelGPU.Name)" -ForegroundColor Green
    Write-Host "Driver Version: $($intelGPU.DriverVersion)" -ForegroundColor Cyan
} else {
    Write-Host "✗ No Intel GPU found" -ForegroundColor Red
    exit 1
}

# Step 3: Set environment variables for Intel GPU
Write-Host "Step 3: Setting Intel GPU environment variables..." -ForegroundColor Yellow
$env:OLLAMA_USE_GPU = "1"
$env:OLLAMA_DEBUG = "1"
$env:OLLAMA_GPU_OVERHEAD = "0"
$env:OLLAMA_MAX_LOADED_MODELS = "1"

# Step 4: Check OpenCL support
Write-Host "Step 4: Checking OpenCL support..." -ForegroundColor Yellow
try {
    $openclVendors = Get-ItemProperty "HKLM:\SOFTWARE\Khronos\OpenCL\Vendors" -ErrorAction SilentlyContinue
    if ($openclVendors) {
        Write-Host "✓ OpenCL vendors found" -ForegroundColor Green
    } else {
        Write-Host "⚠ OpenCL vendors not found - may need Intel GPU drivers" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠ Could not check OpenCL registry" -ForegroundColor Yellow
}

# Step 5: Start Ollama with GPU support
Write-Host "Step 5: Starting Ollama with Intel GPU support..." -ForegroundColor Yellow
Write-Host "Environment variables:" -ForegroundColor Cyan
Write-Host "  OLLAMA_USE_GPU = $env:OLLAMA_USE_GPU" -ForegroundColor Cyan
Write-Host "  OLLAMA_DEBUG = $env:OLLAMA_DEBUG" -ForegroundColor Cyan
Write-Host "  OLLAMA_GPU_OVERHEAD = $env:OLLAMA_GPU_OVERHEAD" -ForegroundColor Cyan

# Start Ollama server
Start-Process -FilePath "ollama" -ArgumentList "serve" -NoNewWindow
Start-Sleep -Seconds 5

# Step 6: Test GPU usage
Write-Host "Step 6: Testing GPU usage..." -ForegroundColor Yellow
$psOutput = & ollama ps
Write-Host "Current models:" -ForegroundColor Cyan
Write-Host $psOutput

if ($psOutput -match "GPU" -or $psOutput -notmatch "100% CPU") {
    Write-Host "✓ SUCCESS: GPU acceleration appears to be working!" -ForegroundColor Green
} else {
    Write-Host "⚠ WARNING: Still showing CPU usage" -ForegroundColor Yellow
    Write-Host "Possible solutions:" -ForegroundColor Yellow
    Write-Host "1. Update Intel Arc GPU drivers" -ForegroundColor White
    Write-Host "2. Install Intel Arc Control software" -ForegroundColor White
    Write-Host "3. Restart computer after driver installation" -ForegroundColor White
    Write-Host "4. Check if Ollama version supports Intel Arc 140V" -ForegroundColor White
}

Write-Host "=== Script completed ===" -ForegroundColor Green
Write-Host "Monitor Task Manager > Performance > GPU for activity during AI inference" -ForegroundColor Cyan
