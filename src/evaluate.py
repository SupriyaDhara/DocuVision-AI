import os
import torch
import pandas as pd
import cv2
from model import CRNN
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_PATH = "data/crnn_dataset/test/test.csv"
MODEL_PATH = "crnn_model_final.pth"
OUTPUT_CSV = "outputs/test_predictions.csv"

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv(CSV_PATH).dropna()
df["text"] = df["text"].astype(str)

# Build charset from training CSV
train_df = pd.read_csv("data/crnn_dataset/train/train.csv").dropna()
train_df["text"] = train_df["text"].astype(str)

all_text = "".join(train_df["text"].values)
chars = sorted(list(set(all_text)))

char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

num_classes = len(char_to_idx) + 1

model = CRNN(num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Model loaded. Starting evaluation...\n")

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

def levenshtein(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        dp[i][0] = i

    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[-1][-1]

def char_accuracy(pred, gt):
    distance = levenshtein(pred, gt)
    return 1 - distance / max(len(gt), 1)

samples = min(500, len(df))
total_acc = 0
results = []

for i in tqdm(range(samples)):
    image_path = df.iloc[i, 0]
    gt_text = df.iloc[i, 1]

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Skipping missing image:", image_path)
        continue

    image = image.astype("float32") / 255.0
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred_text = decode(output)

    acc = char_accuracy(pred_text, gt_text)
    total_acc += acc

    results.append({
        "image_path": image_path,
        "ground_truth": gt_text,
        "prediction": pred_text,
        "character_accuracy": round(acc * 100, 2),
        "cer": round((1 - acc) * 100, 2)
    })

avg_acc = total_acc / len(results)

print("\nCharacter Accuracy (CER-based):", round(avg_acc * 100, 2), "%")
print("CER:", round((1 - avg_acc) * 100, 2), "%")

result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print("Predictions saved to:", OUTPUT_CSV)