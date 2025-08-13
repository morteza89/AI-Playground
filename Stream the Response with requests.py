
#!/usr/bin/env python3
"""
Chat with llama3.2 via Ollama using requests and streaming responses.

Requires:
    pip install requests
"""

import json
import sys
import time
import requests
from typing import List, Dict, Any


OLLAMA_HOST = "http://localhost:11434"   # default Ollama port


def get_model_name():
    model = input("Enter model name (default: llama3.2): ").strip()
    return model if model else "llama3.2"


def chat_stream(messages: List[Dict[str, str]], model_name: str) -> str:
    """
    Send a list of messages to Ollama and stream the assistant's reply in real time.
    Returns the full assistant reply as a string.
    """
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {"model": model_name, "messages": messages, "stream": True}
    try:
        response = requests.post(url, json=payload, stream=True, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f" Error communicating with Ollama: {e}", file=sys.stderr)
        sys.exit(1)

    assistant_reply = ""
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode())
                content = data.get("message", {}).get("content", "")
                assistant_reply += content
                print(content, end="", flush=True)
            except Exception as e:
                print(f"Error parsing chunk: {e}")
                print(line)
    print()  # Newline after streaming reply
    return assistant_reply


def main():
    model_name = get_model_name()
    print(f"Chatting with {model_name} (press Ctrl+C to exit)\n")
    chat_history: List[Dict[str, str]] = []
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            chat_history.append({"role": "user", "content": user_input})
            assistant_reply = chat_stream(chat_history, model_name)
            chat_history.append(
                {"role": "assistant", "content": assistant_reply})
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()


def chat_stream(messages):
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {"model": MODEL_NAME, "messages": messages, "stream": True}
    response = requests.post(url, json=payload, stream=True)

    for line in response.iter_lines():
        if line:
            # Each line is a JSON chunk
            data = json.loads(line.decode())
            # The actual text is in data["message"]["content"]
            print(data["message"]["content"], end="", flush=True)
