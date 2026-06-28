import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from model import CRNN
from dataset import OCRDataset, collate_fn

# ==============================
# DEVICE
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

CSV_PATH = "data/crnn_dataset/train/train.csv"

# ==============================
# LOAD CSV SAFELY
# ==============================
df = pd.read_csv(CSV_PATH)
df = df.dropna()
df["text"] = df["text"].astype(str)

# ==============================
# BUILD CHARACTER SET
# ==============================
all_text = "".join(df["text"].values)
chars = sorted(list(set(all_text)))

char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

num_classes = len(char_to_idx) + 1

print("Total characters:", len(chars))
print("Total training samples:", len(df))

# ==============================
# DATASET + DATALOADER
# ==============================
dataset = OCRDataset(CSV_PATH, char_to_idx)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=True
)

# ==============================
# MODEL
# ==============================
model = CRNN(num_classes).to(device)

# Load previous 30-epoch model
model.load_state_dict(torch.load("crnn_model_final.pth", map_location=device))
print("Previous model loaded successfully")

criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

num_epochs = 5

# Create checkpoint folder
os.makedirs("checkpoints", exist_ok=True)

# ==============================
# TRAINING LOOP
# ==============================
for epoch in range(num_epochs):

    model.train()
    total_loss = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        log_probs = outputs.log_softmax(2)

        input_lengths = torch.full(
            size=(images.size(0),),
            fill_value=log_probs.size(1),
            dtype=torch.long
        ).to(device)

        target_lengths = torch.sum(labels != 0, dim=1)

        loss = criterion(
            log_probs.permute(1, 0, 2),
            labels,
            input_lengths,
            target_lengths
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

    # Save checkpoint after each epoch
    torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1}.pth")

    # Save latest checkpoint
    torch.save(model.state_dict(), "checkpoints/last_model.pth")

    print("Checkpoint saved")

# ==============================
# SAVE FINAL MODEL
# ==============================
torch.save(model.state_dict(), "crnn_model_final.pth")
print("Model saved as crnn_model_final.pth")