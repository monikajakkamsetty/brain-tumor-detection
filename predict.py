"""
predict.py
----------
Loads a trained model and classifies a single new MRI image as
'Tumor' or 'Healthy', printing a confidence percentage.

Usage:
    python predict.py --model brain_tumor_model.keras --image path/to/scan.jpg
"""

import argparse
import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)


def load_image(image_path, img_size=IMG_SIZE):
    img = tf.keras.utils.load_img(image_path, target_size=img_size)
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)  # add batch dimension
    return arr


def predict(model_path, image_path):
    model = tf.keras.models.load_model(model_path)
    img_array = load_image(image_path)

    prob_tumor = float(model.predict(img_array, verbose=0)[0][0])
    label = "Tumor" if prob_tumor >= 0.5 else "Healthy"
    confidence = prob_tumor if label == "Tumor" else (1 - prob_tumor)

    return label, confidence * 100, prob_tumor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="brain_tumor_model.keras", help="Path to trained .keras model")
    parser.add_argument("--image", required=True, help="Path to the MRI image to classify")
    args = parser.parse_args()

    label, confidence, raw_prob = predict(args.model, args.image)

    print(f"\nImage: {args.image}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"(raw P(tumor) = {raw_prob:.4f})")

    if label == "Tumor" and confidence < 65:
        print("\nNote: confidence is relatively low. Consider this a screening "
              "aid, not a diagnosis — recommend clinical follow-up regardless "
              "of the model's output.")


if __name__ == "__main__":
    main()
