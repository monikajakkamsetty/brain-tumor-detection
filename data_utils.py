"""
data_utils.py
-------------
Handles loading and preprocessing of the MRI dataset for binary
classification: Tumor vs Healthy.

Expected folder structure (this is the layout used by most Kaggle /
Hugging Face brain tumor MRI datasets once you unzip them):

    dataset/
        train/
            tumor/
                img001.jpg
                img002.jpg
                ...
            healthy/
                img101.jpg
                ...
        val/                (optional - will be auto-created from train if missing)
            tumor/
            healthy/
        test/               (optional)
            tumor/
            healthy/

If your downloaded dataset uses different class folder names (e.g.
"yes"/"no", "glioma"/"no_tumor", etc.), either rename the folders to
"tumor"/"healthy", or edit CLASS_MAP below.

If the dataset only ships a single folder with all classes mixed
(no train/val/test split), point RAW_DIR at it and run
`prepare_dataset()` once to create a proper split on disk.
"""

import os
import shutil
import random
from pathlib import Path

import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Map whatever folder names your dataset uses -> our two canonical classes.
# Edit this if your Kaggle/HF dataset uses different label names.
CLASS_MAP = {
    "tumor": "tumor",
    "yes": "tumor",
    "glioma": "tumor",
    "meningioma": "tumor",
    "pituitary": "tumor",
    "healthy": "healthy",
    "no": "healthy",
    "no_tumor": "healthy",
    "notumor": "healthy",
}


def prepare_dataset(raw_dir: str, output_dir: str, val_split=0.15, test_split=0.15):
    """
    One-time utility: takes a raw dataset directory where each
    sub-folder is a class (e.g. 'yes', 'no', 'glioma', ...), maps
    those classes into 'tumor' / 'healthy', and writes a proper
    train/val/test split to `output_dir`.

    Only needed if your downloaded dataset isn't already split into
    train/val/test folders.
    """
    random.seed(SEED)
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    for split in ("train", "val", "test"):
        for cls in ("tumor", "healthy"):
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    for class_folder in raw_dir.iterdir():
        if not class_folder.is_dir():
            continue
        canonical = CLASS_MAP.get(class_folder.name.lower())
        if canonical is None:
            print(f"Skipping unrecognized folder: {class_folder.name} "
                  f"(add it to CLASS_MAP in data_utils.py if it's a valid class)")
            continue

        images = [f for f in class_folder.iterdir() if f.is_file()]
        random.shuffle(images)

        n = len(images)
        n_val = int(n * val_split)
        n_test = int(n * test_split)

        val_imgs = images[:n_val]
        test_imgs = images[n_val:n_val + n_test]
        train_imgs = images[n_val + n_test:]

        for split_name, split_imgs in (("train", train_imgs), ("val", val_imgs), ("test", test_imgs)):
            for img_path in split_imgs:
                dest = output_dir / split_name / canonical / img_path.name
                shutil.copy2(img_path, dest)

        print(f"{class_folder.name} -> {canonical}: "
              f"{len(train_imgs)} train / {len(val_imgs)} val / {len(test_imgs)} test")

    print(f"\nDataset prepared at: {output_dir}")


def get_datasets(data_dir: str, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Loads train/val/test datasets from an already-split directory
    (data_dir/train, data_dir/val, data_dir/test), each containing
    'tumor' and 'healthy' sub-folders.

    Returns (train_ds, val_ds, test_ds, class_names) as tf.data.Dataset
    objects, already batched, shuffled (train only), and prefetched.
    Pixel values are scaled to [0, 1] via a Rescaling layer inside the
    model itself (see model.py), so raw images are returned here.
    """
    data_dir = Path(data_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train",
        labels="inferred",
        label_mode="binary",
        class_names=["healthy", "tumor"],  # healthy=0, tumor=1
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=SEED,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "val",
        labels="inferred",
        label_mode="binary",
        class_names=["healthy", "tumor"],
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )

    test_dir = data_dir / "test"
    test_ds = None
    if test_dir.exists():
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            labels="inferred",
            label_mode="binary",
            class_names=["healthy", "tumor"],
            image_size=img_size,
            batch_size=batch_size,
            shuffle=False,
        )

    class_names = ["healthy", "tumor"]

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    if test_ds is not None:
        test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
