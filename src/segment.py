import cv2
import os
import shutil


def crop_text_lines(image_path, output_folder="outputs/cropped_lines"):
    preview_folder = "static/previews"

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    if os.path.exists(preview_folder):
        shutil.rmtree(preview_folder)

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(preview_folder, exist_ok=True)

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found")

    # Save original preview
    original_preview = os.path.join(preview_folder, "original.png")
    cv2.imwrite(original_preview, image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Save grayscale preview
    grayscale_preview = os.path.join(preview_folder, "grayscale.png")
    cv2.imwrite(grayscale_preview, gray)

    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    # Save threshold preview
    threshold_preview = os.path.join(preview_folder, "threshold.png")
    cv2.imwrite(threshold_preview, binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (90, 4))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    lines = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if h < 12 or w < 80:
            continue

        if h > 80:
            continue

        padding_x = 8
        padding_y = 6

        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(image.shape[1], x + w + padding_x)
        y2 = min(image.shape[0], y + h + padding_y)

        crop = image[y1:y2, x1:x2]

        lines.append((y1, x1, crop))

    lines.sort(key=lambda item: (item[0], item[1]))

    cropped_paths = []
    preview_line_paths = []

    for i, (_, _, crop) in enumerate(lines):
        path = os.path.join(output_folder, f"line_{i}.png")
        cv2.imwrite(path, crop)
        cropped_paths.append(path)

        preview_path = os.path.join(preview_folder, f"line_{i}.png")
        cv2.imwrite(preview_path, crop)
        preview_line_paths.append("/" + preview_path.replace("\\", "/"))

    preview_paths = {
        "original": "/static/previews/original.png",
        "grayscale": "/static/previews/grayscale.png",
        "threshold": "/static/previews/threshold.png",
        "lines": preview_line_paths
    }

    return cropped_paths, preview_paths