"""
train.py
--------
Trains the brain tumor classifier and saves the best model to disk.

Usage:
    python train.py --data_dir dataset --model_type transfer --epochs 25

Arguments:
    --data_dir     Path to dataset with train/ val/ (test/ optional) subfolders,
                    each containing 'tumor' and 'healthy' subfolders.
    --model_type    'custom' (CNN from scratch) or 'transfer' (EfficientNetB0). Default: transfer
    --epochs        Number of training epochs. Default: 25
    --batch_size    Batch size. Default: 32
    --output        Where to save the trained model. Default: brain_tumor_model.keras
"""

import argparse
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import tensorflow as tf

from data_utils import get_datasets, IMG_SIZE
from model import build_custom_cnn, build_transfer_model, compile_model


def plot_history(history, out_path="training_history.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="train_loss")
    axes[0].plot(history.history["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train_acc")
    axes[1].plot(history.history["val_accuracy"], label="val_acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved training curves to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to dataset root (with train/val/test)")
    parser.add_argument("--model_type", choices=["custom", "transfer"], default="transfer")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", default="brain_tumor_model.keras")
    args = parser.parse_args()

    print("Loading datasets...")
    train_ds, val_ds, test_ds, class_names = get_datasets(
        args.data_dir, img_size=IMG_SIZE, batch_size=args.batch_size
    )
    print(f"Classes: {class_names} (0=healthy, 1=tumor)")

    print(f"Building {args.model_type} model...")
    if args.model_type == "custom":
        model = build_custom_cnn(input_shape=IMG_SIZE + (3,))
        lr = 1e-3
    else:
        model = build_transfer_model(input_shape=IMG_SIZE + (3,))
        lr = 1e-4

    model = compile_model(model, learning_rate=lr)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.output, monitor="val_auc", mode="max", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=6, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
        ),
    ]

    # Handle class imbalance automatically (common in medical datasets)
    healthy_count = sum(1 for _, y in train_ds.unbatch() if y.numpy()[0] == 0)
    tumor_count = sum(1 for _, y in train_ds.unbatch() if y.numpy()[0] == 1)
    total = healthy_count + tumor_count
    class_weight = {
        0: total / (2.0 * healthy_count) if healthy_count else 1.0,
        1: total / (2.0 * tumor_count) if tumor_count else 1.0,
    }
    print(f"Class balance -> healthy: {healthy_count}, tumor: {tumor_count}")
    print(f"Using class weights: {class_weight}")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    plot_history(history)

    if test_ds is not None:
        print("\nEvaluating on test set:")
        results = model.evaluate(test_ds, return_dict=True)
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")

    print(f"\nBest model saved to: {args.output}")


if __name__ == "__main__":
    main()
