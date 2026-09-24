"""
Test the trained model against an arbitrary external image file (not an
MNIST sample) -- e.g. a photo or scan of a handwritten digit.

Usage: python3 src/test_downloaded_image.py <path-to-image> [--no-display]
"""

import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

MODEL_PATH = "models/lenet5_relu.keras"


def otsu_threshold(gray):
    """Standard Otsu's-method threshold pick, implemented directly since
    this project has no OpenCV dependency. `gray` is a 0-255 float array."""
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b, weight_b, best_var, threshold = 0.0, 0.0, 0.0, 0
    for t in range(256):
        weight_b += hist[t]
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += t * hist[t]
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        between_var = weight_b * weight_f * (mean_b - mean_f) ** 2
        if between_var > best_var:
            best_var, threshold = between_var, t
    return threshold


def load_and_preprocess(image_path):
    img = Image.open(image_path).convert("L")           # grayscale
    raw = np.array(img).astype("float32")                # 0-255, original resolution

    # MNIST digits are white strokes on a black background. A photographed/
    # downloaded digit is usually the opposite (dark ink on light paper), so
    # naively reusing MNIST's polarity would feed the model an image it was
    # never trained to see. Detect it from the border pixels (background)
    # vs the image mean, and invert if the border is the brighter side.
    border = np.concatenate([raw[0, :], raw[-1, :], raw[:, 0], raw[:, -1]])
    inverted = False
    if border.mean() > raw.mean():
        raw = 255.0 - raw
        inverted = True

    # A real photo also carries JPEG noise, colored ink, and uneven paper
    # lighting that a clean MNIST bitmap never has. Otsu's method picks a
    # single background/foreground cutoff from the image's own histogram
    # and binarizing to it strips all of that out, leaving a clean stroke
    # silhouette much closer to what the model was actually trained on.
    threshold = otsu_threshold(raw)
    binary = np.where(raw > threshold, 255.0, 0.0)

    # A plain resize-to-28x28 leaves the digit however large/wherever
    # positioned it was in the original photo, which almost never matches
    # MNIST's own preprocessing (every MNIST digit is bounding-box-cropped
    # and centered so the stroke fills most of the frame). Reproduce that:
    # find the ink's bounding box, crop to it, then resize the crop to fit
    # a 20x20 box (preserving aspect ratio, like the real MNIST pipeline)
    # and center it on a 28x28 canvas.
    rows = np.where(binary.any(axis=1))[0]
    cols = np.where(binary.any(axis=0))[0]
    if len(rows) > 0 and len(cols) > 0:
        cropped = binary[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    else:
        cropped = binary  # blank/uniform image -- nothing to crop, use as-is

    crop_img = Image.fromarray(cropped.astype("uint8"))
    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    crop_img = crop_img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    canvas.paste(crop_img, (paste_x, paste_y))

    arr = np.array(canvas).astype("float32") / 255.0     # same normalization as train.py
    return arr, inverted


def main():
    parser = argparse.ArgumentParser(description="Classify an external handwritten-digit image.")
    parser.add_argument("image_path", help="Path to the image file to classify")
    parser.add_argument("--no-display", action="store_true", help="Skip the matplotlib window")
    args = parser.parse_args()

    model = tf.keras.models.load_model(MODEL_PATH)

    arr, inverted = load_and_preprocess(args.image_path)
    model_input = arr.reshape(1, 28, 28, 1)

    probs = model.predict(model_input, verbose=0)[0]
    predicted_label = np.argmax(probs)
    confidence = probs[predicted_label] * 100

    print(f"Image            : {args.image_path}")
    print(f"Auto-inverted    : {'YES (dark-on-light input detected)' if inverted else 'NO (already light-on-dark)'}")
    print(f"Predicted label  : {predicted_label}")
    print(f"Confidence       : {confidence:.2f}%")

    print("\nFull probability breakdown:")
    for digit in range(10):
        bar = "#" * int(probs[digit] * 50)
        print(f"  {digit}: {probs[digit]*100:5.2f}%  {bar}")

    if not args.no_display:
        plt.figure(figsize=(4, 4))
        plt.imshow(arr, cmap="gray")
        plt.title(f"Predicted: {predicted_label} ({confidence:.1f}%)", color="blue")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
