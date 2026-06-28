import os
import sys
import re
import torch
import cv2
import pandas as pd

sys.path.append(os.path.dirname(__file__))

from model import CRNN
from segment import crop_text_lines

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_PATH = "data/crnn_dataset/train/train.csv"
MODEL_PATH = "crnn_model_final.pth"

df = pd.read_csv(CSV_PATH).dropna()
df["text"] = df["text"].astype(str)

all_text = "".join(df["text"].values)
chars = sorted(list(set(all_text)))

char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

num_classes = len(char_to_idx) + 1

model = CRNN(num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Model loaded successfully.")


def decode(pred):
    pred = pred.argmax(2)
    pred = pred.squeeze(0)

    decoded = []
    prev = -1

    for p in pred:
        p = p.item()
        if p != 0 and p != prev:
            decoded.append(idx_to_char.get(p, ""))
        prev = p

    return "".join(decoded)


def clean_ocr_text(text):
    # Remove unusual OCR symbols but keep useful punctuation
    text = re.sub(r"[^a-zA-Z0-9\s.,:;!?'\-]", "", text)

    # Remove repeated spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Clean spacing around new lines
    text = re.sub(r"\n\s+", "\n", text)

    return text.strip()


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Image not found. Check image path.")

    h, w = image.shape

    new_h = 32
    new_w = int(w * (new_h / h))

    min_w = 128
    if new_w < min_w:
        new_w = min_w

    image = cv2.resize(image, (new_w, new_h))
    image = image.astype("float32") / 255.0

    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)

    return image.to(device)


def predict_text(image_path):
    image = preprocess_image(image_path)

    with torch.no_grad():
        output = model(image)
        prediction = decode(output)

    return prediction


def predict_full_document(image_path):
    line_paths, preview_paths = crop_text_lines(image_path)

    final_text = []

    for line_path in line_paths:
        try:
            text = predict_text(line_path)

            if text.strip():
                final_text.append(text)

        except Exception as e:
            print(f"Skipping {line_path} due to error: {e}")

    if len(final_text) == 0:
        return "No readable text lines detected."

    raw_text = "\n".join(final_text)
    cleaned_text = clean_ocr_text(raw_text)

    return cleaned_text, preview_paths