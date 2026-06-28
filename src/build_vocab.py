import pandas as pd
import json

labels_path = "data/crnn_dataset/train/labels.csv"

df = pd.read_csv(labels_path)

all_text = "".join(df["text"].astype(str).values)

vocab = sorted(list(set(all_text)))

# CTC blank token add
vocab.append("_")

char_to_idx = {char: idx for idx, char in enumerate(vocab)}

print("Vocabulary size:", len(vocab))

# Save vocab
with open("data/crnn_dataset/vocab.json", "w") as f:
    json.dump(char_to_idx, f)

print("vocab.json saved successfully!")






