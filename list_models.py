import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the API with your API key
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# List available models
print("Available models:")
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"- {model.name} (supports generateContent)")

print("\nNote: Use one of these model names in your configuration.")
