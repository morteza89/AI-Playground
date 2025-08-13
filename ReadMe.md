# Ollama Python Chat Playground

This project demonstrates three different ways to chat with a local Ollama server running the llama3.2 model using Python.

## Project Setup Instructions

1. **Install Ollama**

   - Download and install from [Ollama website](https://ollama.com/download)

2. **Start the Ollama server**

   - Open a terminal and run:
     ```
     ollama serve
     ```

3. **Pull the llama3.2 model**

   - Run:
     ```
     ollama pull llama3.2
     ```

4. **Set up Python environment**

   - Create a virtual environment (recommended):
     ```
     uv venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - Install required packages:
     ```
     uv pip install requests ollama
     ```

5. **Run any of the scripts**
   - Example:
     ```
     python "Using requests (no external Ollama client).py"
     python "Using_Official_ollama_pt_Client.py"
     python "Stream the Response with requests.py"
     ```

## Model Selection Feature

All three scripts now support easy model selection at runtime. When you start any script, you will be prompted to enter the model name you want to use (e.g., `llama3.2`, `gpt-oss:20b`, `qwen3:4b`, etc.).

- If you press Enter without typing a model name, the default (`llama3.2`) will be used.
- Make sure the model is pulled and available in Ollama (`ollama list` to see available models, `ollama pull <modelname>` to download).

This makes it simple to experiment and chat with any model you have installed in Ollama, without editing the scripts.

## How It Works: Local vs Cloud-Based Models

- **Local**: All scripts in this project connect to a locally running Ollama server (`http://localhost:11434`). The model runs on your laptop's CPU or GPU (if configured). No data leaves your machine except for the initial model download.

- **Cloud-Based**: If you use a cloud-hosted Ollama server (not covered here), you would change the host address to point to the remote server. In that case, your prompts and responses are processed remotely.

**Note:** By default, Ollama runs locally. The model is loaded and executed on your hardware. You can configure Ollama to use your CPU, iGPU, or GPU by setting environment variables before starting the server (see Ollama docs for details).

## Script Comparison: Three Approaches

### 1. Using requests (no external Ollama client).py

- Uses the plain HTTP API via the `requests` library.
- Manually parses streaming JSON responses from Ollama.
- Gives you more control over the raw HTTP response and error handling.
- Requires only the `requests` package.
- Can be customized for any HTTP API changes.

### 2. Using_Official_ollama_pt_Client.py

- Uses the official Ollama Python client (`ollama` package).
- Handles streaming and non-streaming responses with built-in methods.
- Simpler and more readable code, abstracts away HTTP details.
- May support more Ollama features and future updates.
- Requires the `ollama` package.

### 3. Stream the Response with requests.py

- Uses the `requests` library and sets `stream=True` in the payload.
- Streams the assistant's reply in real time, printing each chunk as it arrives.
- Useful for interactive applications where immediate feedback is desired.
- Requires only the `requests` package.

## Hardware Usage

- By default, Ollama runs on your CPU.
- To use your Intel iGPU or NVIDIA GPU, set the environment variable before starting Ollama:
  - Intel iGPU: `$env:OLLAMA_USE_GPU="1"`
  - NVIDIA GPU: Ollama uses CUDA automatically if available.
- Start Ollama after setting the variable to enable GPU acceleration.

## Troubleshooting

- If you see `Error: listen tcp 127.0.0.1:11434: bind: Only one usage of each socket address...`, Ollama is already running. Stop the existing process before starting a new one.
- If a script fails to connect, make sure Ollama is running and the model is pulled.
- For GPU issues, check Ollama logs and ensure your drivers are up to date.

---

Feel free to use any of the three scripts to chat with your local Ollama model. Each approach offers different trade-offs in terms of control, simplicity, and streaming support.
