# Ollama Python Chat Playground with different LLM models

This project demonstrates four different ways to chat with a local Ollama server running models like GPT-oss-20b, llama3.2, Qwen3, etc. using Python.

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
     python "Ollama_GUI_Chat.py"
     ```

## Model Selection Feature

All three scripts now support easy model selection at runtime. When you start any script, you will be prompted to enter the model name you want to use (e.g., `llama3.2`, `gpt-oss:20b`, `qwen3:4b`, etc.).

- If you press Enter without typing a model name, the default (`llama3.2`) will be used.
- Make sure the model is pulled and available in Ollama (`ollama list` to see available models, `ollama pull <modelname>` to download).
- The GUI version (`Ollama_GUI_Chat.py`) provides a dropdown menu for model selection and switching.

This makes it simple to experiment and chat with any model you have installed in Ollama, without editing the scripts.

## How It Works: Local vs Cloud-Based Models

- **Local**: All scripts in this project connect to a locally running Ollama server (`http://localhost:11434`). The model runs on your laptop's CPU or GPU (if configured). No data leaves your machine except for the initial model download.

- **Cloud-Based**: If you use a cloud-hosted Ollama server (not covered here), you would change the host address to point to the remote server. In that case, your prompts and responses are processed remotely.

**Note:** By default, Ollama runs locally. The model is loaded and executed on your hardware. You can configure Ollama to use your CPU, iGPU, or GPU by setting environment variables before starting the server (see Ollama docs for details).

## Script Comparison: Four Approaches

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

### 4. Ollama_GUI_Chat.py

- Provides a full graphical user interface using `tkinter`.
- Features tabbed interface with chat, model settings, and hardware configuration.
- Real-time model switching without restarting the application.
- Advanced hardware detection and GPU configuration tools.
- Built-in Intel iGPU optimization and troubleshooting features.
- Requires the `ollama` package and `tkinter` (included with Python).

## Detailed Comparison: Pros and Cons

| Feature               | Using requests (no external Ollama client) | Using Official ollama Client         | Stream the Response with requests   | Ollama_GUI_Chat.py                  |
| --------------------- | ------------------------------------------ | ------------------------------------ | ----------------------------------- | ----------------------------------- |
| **Dependencies**      | ✅ Only `requests` (minimal)               | ⚠️ Requires `ollama` package         | ✅ Only `requests` (minimal)        | ⚠️ Requires `ollama` + `tkinter`    |
| **Code Complexity**   | ⚠️ More verbose, manual JSON parsing       | ✅ Simple, clean, readable           | ⚠️ Moderate, manual stream handling | ⚠️ Complex GUI, many features       |
| **Error Handling**    | ✅ Full control over HTTP errors           | ⚠️ Abstracted, less granular control | ⚠️ Manual error handling needed     | ✅ Comprehensive error handling     |
| **Streaming Support** | ❌ No real-time streaming                  | ✅ Built-in streaming methods        | ✅ Real-time token streaming        | ✅ Built-in streaming in GUI        |
| **Future-Proofing**   | ⚠️ Manual updates for API changes          | ✅ Automatic updates with package    | ⚠️ Manual updates for API changes   | ✅ Official client + GUI framework  |
| **Performance**       | ✅ Direct HTTP, minimal overhead           | ⚠️ Additional abstraction layer      | ✅ Efficient streaming, low latency | ⚠️ GUI overhead, but good UX        |
| **Customization**     | ✅ Full control over requests              | ⚠️ Limited to client capabilities    | ✅ Full control over streaming      | ✅ Highly customizable interface    |
| **Learning Value**    | ✅ Understand HTTP/JSON internals          | ⚠️ Less insight into underlying API  | ✅ Learn streaming protocols        | ✅ Learn GUI development + AI APIs  |
| **Production Use**    | ⚠️ Requires more error handling            | ✅ Battle-tested, robust             | ✅ Good for real-time apps          | ✅ Professional desktop application |
| **Debugging**         | ✅ Easy to inspect raw responses           | ⚠️ Harder to debug client issues     | ✅ Can monitor stream chunks        | ✅ GUI feedback + debugging tools   |
| **User Experience**   | ❌ Command-line only                       | ❌ Command-line only                 | ❌ Command-line only                | ✅ Intuitive graphical interface    |
| **Hardware Control**  | ❌ No built-in GPU configuration           | ❌ No built-in GPU configuration     | ❌ No built-in GPU configuration    | ✅ Advanced GPU detection & setup   |

### When to Use Each Method:

**🔧 Use requests (no external client) when:**

- You want minimal dependencies
- You need full control over HTTP requests
- You're building a custom wrapper
- You want to understand the API internals
- You're working in constrained environments

**📦 Use Official ollama Client when:**

- You want the simplest, most maintainable code
- You need robust error handling out of the box
- You want automatic updates and new features
- You're building production applications quickly
- You prefer official, supported solutions

**⚡ Use Stream with requests when:**

- You need real-time response streaming
- You're building interactive chat interfaces
- You want immediate user feedback
- You need to process responses token-by-token
- You want minimal dependencies with streaming

**🖥️ Use Ollama_GUI_Chat.py when:**

- You want a professional desktop chat application
- You need easy model switching and management
- You want advanced hardware configuration tools
- You're building apps for non-technical users
- You need Intel iGPU optimization and troubleshooting
- You prefer point-and-click over command-line interfaces

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

Feel free to use any of the four scripts to chat with your local Ollama model. Each approach offers different trade-offs in terms of control, simplicity, streaming support, and user experience. The GUI version provides the most user-friendly experience, while the command-line versions offer more technical control and learning opportunities.
