import os
import json
import cv2
import csv
import shutil

# ==============================
# INPUT FOLDERS
# ==============================
IMAGE_FOLDER = "data/document/test/images"
ANNOTATION_FOLDER = "data/document/test/annotations"

# ==============================
# OUTPUT FOLDERS
# ==============================
OUTPUT_FOLDER = "data/crnn_dataset/test"
OUTPUT_IMAGE_FOLDER = os.path.join(OUTPUT_FOLDER, "images")
OUTPUT_LABEL_FILE = os.path.join(OUTPUT_FOLDER, "test.csv")

# ==============================
# CLEAN OLD TEST DATA
# ==============================
if os.path.exists(OUTPUT_FOLDER):
    shutil.rmtree(OUTPUT_FOLDER)

os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)

# ==============================
# PREPROCESS FUNCTION
# ==============================
def preprocess_image(cropped):
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    h, w = cropped.shape

    if h == 0 or w == 0:
        return None

    new_h = 32
    new_w = int(w * (32 / h))

    if new_w <= 0:
        return None

    resized = cv2.resize(cropped, (new_w, new_h))

    return resized

# ==============================
# MAIN FUNCTION
# ==============================
def prepare_dataset():
    total_saved = 0

    with open(OUTPUT_LABEL_FILE, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_path", "text"])

        for json_file in os.listdir(ANNOTATION_FOLDER):

            if not json_file.endswith(".json"):
                continue

            json_path = os.path.join(ANNOTATION_FOLDER, json_file)

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            image_id = data["form"]["id"]
            image_name = image_id + ".png"
            image_path = os.path.join(IMAGE_FOLDER, image_name)

            if not os.path.exists(image_path):
                continue

            image = cv2.imread(image_path)

            if image is None:
                continue

            for idx, line in enumerate(data["lines"]):

                text = line["text"].strip()

                if text == "":
                    continue

                if len(text) < 3:
                    continue

                if not any(c.isalpha() for c in text):
                    continue

                bbox = line["bounding_box"]
                x1, y1 = bbox["x1"], bbox["y1"]
                x2, y2 = bbox["x2"], bbox["y2"]

                cropped = image[y1:y2, x1:x2]

                if cropped is None or cropped.size == 0:
                    continue

                processed = preprocess_image(cropped)

                if processed is None:
                    continue

                output_filename = f"{image_id}_{idx}.png"
                output_path = os.path.join(OUTPUT_IMAGE_FOLDER, output_filename)

                cv2.imwrite(output_path, processed)

                clean_path = output_path.replace("\\", "/")

                writer.writerow([clean_path, text])

                total_saved += 1

    print("\n==============================")
    print("Test dataset preparation complete!")
    print("Total clean test lines saved:", total_saved)
    print("Test CSV saved at:", OUTPUT_LABEL_FILE)
    print("==============================\n")

if __name__ == "__main__":
    prepare_dataset()