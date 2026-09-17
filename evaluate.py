"""
evaluate.py
-----------
Runs the trained model over the test set and prints/saves a
confusion matrix and classification report (precision/recall/F1
per class) -- important for medical models where false negatives
(missed tumors) matter more than raw accuracy.

Usage:
    python evaluate.py --model brain_tumor_model.keras --data_dir dataset
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from data_utils import get_datasets, IMG_SIZE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="brain_tumor_model.keras")
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    _, _, test_ds, class_names = get_datasets(args.data_dir, img_size=IMG_SIZE)

    if test_ds is None:
        raise SystemExit("No test/ folder found under data_dir.")

    y_true, y_prob = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0).flatten()
        y_prob.extend(preds.tolist())
        y_true.extend(labels.numpy().flatten().tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Saved confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    print("Saved roc_curve.png")


if __name__ == "__main__":
    main()
