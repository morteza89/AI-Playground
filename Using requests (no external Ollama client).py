#!/usr/bin/env python3
"""
Chat with llama3.2 via Ollama using the plain HTTP API.

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


def chat_with_ollama(messages: List[Dict[str, str]]) -> str:
    """
    Send a list of messages to Ollama and return the assistant's reply.
    Each message dict must contain 'role' ('user' or 'assistant') and 'content'.
    """
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        # "stream": False  # set to True to get chunked streaming
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error communicating with Ollama: {e}", file=sys.stderr)
        sys.exit(1)

    # Ollama streams multiple JSON objects, one per line
    assistant_reply = ""
    for line in response.iter_lines():
        if line:
            try:
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                assistant_reply += content
            except Exception as e:
                print(f"Error parsing chunk: {e}")
                print(line)
    return assistant_reply


def main():
    model_name = get_model_name()
    print(f"Chatting with {model_name} (press Ctrl+C to exit)\n")
    chat_history: List[Dict[str, str]] = []
    global MODEL_NAME
    MODEL_NAME = model_name
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            chat_history.append({"role": "user", "content": user_input})
            assistant_reply = chat_with_ollama(chat_history)
            chat_history.append(
                {"role": "assistant", "content": assistant_reply})
            print(f"{model_name}: {assistant_reply}\n")
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
