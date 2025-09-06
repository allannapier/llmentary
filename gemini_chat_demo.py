import os
import sys
from llmentary import monitor, AutoInstrument

# Instrumentation: just 3 lines
AutoInstrument.auto_patch_all()
monitor.configure(store_raw_text=False)  # Privacy-first

# Gemini API key (set this as an environment variable)
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set.")
    print("Please set your API key: export GEMINI_API_KEY=your_api_key_here")
    sys.exit(1)

try:
    import google.generativeai as genai
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
except ImportError:
    print("Google Generative AI library not installed.")
    print("Install it with: pip install google-generativeai")
    sys.exit(1)

def chat():
    print("Welcome to llmentary Gemini Chat!")
    print("Type 'exit' to quit.")
    print("All conversations are automatically monitored for drift detection.")
    print()
    
    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
            
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        try:
            # Generate response
            response = model.generate_content(user_input)
            output_text = response.text
            
            # Manually record the call since Gemini isn't auto-patched yet
            monitor.record_call(
                input_text=user_input,
                output_text=output_text,
                model="gemini-2.0-flash-exp",
                provider="google"
            )
            
            print("Gemini:", output_text)
            
        except Exception as e:
            print(f"Error: {e}")
            print("Please check your API key and internet connection.")

if __name__ == "__main__":
    chat()
