import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np


class OCRDataset(Dataset):
    def __init__(self, csv_file, char_to_idx):
        self.data = pd.read_csv(csv_file).dropna()
        self.data["text"] = self.data["text"].astype(str)
        self.char_to_idx = char_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]
        text = self.data.iloc[idx, 1]

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = image.astype("float32") / 255.0

        image = torch.tensor(image).unsqueeze(0)  # (1, H, W)

        label = [self.char_to_idx[c] for c in text]
        label = torch.tensor(label, dtype=torch.long)

        return image, label


# ==============================
# CUSTOM COLLATE FUNCTION
# ==============================
def collate_fn(batch):

    images, labels = zip(*batch)

    # -------- PAD IMAGES (WIDTH DIMENSION) --------
    heights = [img.shape[1] for img in images]
    widths = [img.shape[2] for img in images]

    max_width = max(widths)

    padded_images = []
    for img in images:
        pad_width = max_width - img.shape[2]

        padded = torch.nn.functional.pad(
            img,
            (0, pad_width, 0, 0),  # pad only width
            value=0
        )
        padded_images.append(padded)

    images = torch.stack(padded_images)

    # -------- PAD LABELS --------
    labels = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=0
    )

    return images, labels
