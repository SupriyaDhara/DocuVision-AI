import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
print("API Key Loaded:", API_KEY is not None)
print(API_KEY[:10] if API_KEY else "No Key")

if API_KEY:
    genai.configure(api_key=API_KEY)

def generate_summary(text):
    if not API_KEY:
        return "Gemini API key not found. Please check your .env file."

    if not text or text.strip() == "":
        return "No text available for summary."

    model = genai.GenerativeModel("models/gemini-2.5-flash")

    prompt = f"""
    Summarize the following OCR extracted document text in simple English.
    Keep it short and clear.

    Text:
    {text}
    """

    response = model.generate_content(prompt)

    return response.text