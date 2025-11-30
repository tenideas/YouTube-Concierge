from __future__ import annotations

import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables.
load_dotenv()

def main():
    '# Script entry point.'
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        sys.exit(1)

    genai.configure(api_key=api_key)

    print("Fetching available models...\n")
    
    try:
        # List models from the API.
        for model in genai.list_models():
            # Filter for models that support content generation.
            if "generateContent" in model.supported_generation_methods:
                print(f"Name: {model.name}")
                print(f"  Display Name: {model.display_name}")
                print(f"  Input Token Limit: {model.input_token_limit}")
                print(f"  Output Token Limit: {model.output_token_limit}")
                print("-" * 40)
                
    except Exception as e:
        print(f"Failed to list models: {e}")

if __name__ == "__main__":
    main()
    
