#!/usr/bin/env python3
"""
Chat with llama3.2 via Ollama using the official Python client.

Requires:
    pip install ollama
"""

import sys
import time
from ollama import Client


def get_model_name():
    model = input("Enter model name (default: llama3.2): ").strip()
    return model if model else "llama3.2"


def main():
    model_name = get_model_name()
    client = Client(host="http://localhost:11434")
    print(f"Chatting with {model_name} (press Ctrl+C to exit)\n")
    chat_history = []
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            chat_history.append({"role": "user", "content": user_input})
            response = client.chat(
                model=model_name,
                messages=chat_history,
                stream=False,
            )
            assistant_reply = response.get("message", {}).get("content", "")
            print(assistant_reply)
            chat_history.append(
                {"role": "assistant", "content": assistant_reply})
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
