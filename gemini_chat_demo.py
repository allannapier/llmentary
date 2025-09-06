import os
import sys
from llmentary import monitor
from llmentary.interceptors import AutoInstrument
from google.generativeai import GenerativeModel, configure

# Instrumentation: just 3 lines
AutoInstrument.auto_patch_all()
monitor.configure(store_raw_text=False)  # Privacy-first

# Gemini API key (set this as an environment variable or pass later)
API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")

if API_KEY == "YOUR_API_KEY_HERE":
    print("Please set your Gemini API key in the GEMINI_API_KEY environment variable.")
    sys.exit(1)

configure(api_key=API_KEY)
model = GenerativeModel("gemini-2.0-flash")

def chat():
    print("Welcome to llmentary Gemini Chat!")
    print("Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = model.generate_content(user_input)
        print("Gemini:", response.text)

if __name__ == "__main__":
    chat()
