import os
import requests

MODEL_URL = "https://huggingface.co/supriyadhara03/DocuVision-CRNN-Model/resolve/main/crnn_model_final.pth"
MODEL_PATH = "crnn_model_final.pth"

def download_model():
    if os.path.exists(MODEL_PATH):
        print("Model already exists.")
        return

    print("Downloading model...")

    response = requests.get(MODEL_URL)
    response.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

    print("Model downloaded successfully.")