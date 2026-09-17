# Create a virtual environment

run: python -m venv venv.
then activate it — Windows:
venv\Scripts\activate
# Brain Tumor Detection from MRI — CNN Classifier

A complete pipeline for binary classification of brain MRI scans:
**Tumor** vs **Healthy**, with a confidence percentage on each prediction.

## 1. Get the dataset

Pick one:

**Option A — Kaggle (Brain Tumor Classification MRI)**
```bash
pip install kaggle
# place your kaggle.json API token in ~/.kaggle/ first
kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri
unzip brain-tumor-classification-mri.zip -d raw_dataset
```

**Option B — Hugging Face**
```bash
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('sartajbhuvaji/brain-tumor-classification')
ds.save_to_disk('raw_dataset')
"
```
(exact dataset name may vary — search "brain tumor MRI" on
huggingface.co/datasets and adjust the `load_dataset` call. You'll then
need to export the images to folders per class to match the structure below.)

## 2. Arrange the data

This project expects:
```
dataset/
  train/
    tumor/
    healthy/
  val/
    tumor/
    healthy/
  test/
    tumor/
    healthy/
```

Most Kaggle brain tumor datasets ship with classes like `yes`/`no` or
`glioma`/`meningioma`/`pituitary`/`no_tumor`. If your raw download is a
single folder of class-subfolders with no split, run:

```bash
python -c "
from data_utils import prepare_dataset
prepare_dataset('raw_dataset/Training', 'dataset')
"
```
This maps any of `yes/glioma/meningioma/pituitary` → `tumor` and
`no/no_tumor/healthy` → `healthy` (edit `CLASS_MAP` in `data_utils.py`
if your folder names differ), and writes a proper 70/15/15 train/val/test
split into `dataset/`.

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Train

```bash
python train.py --data_dir dataset --model_type transfer --epochs 25
```
- `--model_type transfer` (default) uses an ImageNet-pretrained
  EfficientNetB0 backbone, fine-tuned — this gives the best accuracy
  with a few thousand images and is the recommended option.
- `--model_type custom` trains a CNN from scratch (`model.py ->
  build_custom_cnn`) — useful for learning how each layer works, but
  needs more data/epochs to reach comparable accuracy.

The script automatically:
- balances the loss for class imbalance (common in medical datasets),
- augments training images (flips/rotation/zoom) without touching val/test,
- saves the best checkpoint (by validation AUC) to `brain_tumor_model.keras`,
- plots loss/accuracy curves to `training_history.png`,
- reports final test-set metrics if a `test/` folder exists.

## 5. Evaluate

```bash
python evaluate.py --model brain_tumor_model.keras --data_dir dataset
```
Outputs a classification report (precision/recall/F1 per class),
`confusion_matrix.png`, and `roc_curve.png`. For a medical screening
model, **recall on the tumor class** (i.e. minimizing missed tumors /
false negatives) matters more than raw accuracy — check that number
specifically.

## 6. Predict on a new image

```bash
python predict.py --model brain_tumor_model.keras --image path/to/new_scan.jpg
```
Example output:
```
Image: path/to/new_scan.jpg
Prediction: Tumor
Confidence: 94.32%
(raw P(tumor) = 0.9432)
```

## How it works

- **Input**: MRI image resized to 224×224×3.
- **Backbone**: EfficientNetB0 (transfer learning) or a 4-block custom
  CNN, both ending in global average pooling → dense layers → a single
  sigmoid output representing P(tumor).
- **Label rule**: `Tumor` if P(tumor) ≥ 0.5, else `Healthy`.
- **Confidence**: `max(P(tumor), 1 - P(tumor))` as a percentage — i.e.
  how far the prediction sits from the 50/50 decision boundary.
- **Loss**: binary cross-entropy, with automatic class weighting.
- **Regularization**: dropout + data augmentation to reduce overfitting,
  since medical imaging datasets are usually small (a few thousand images).

## Important note

This is a research/educational pipeline, not a certified diagnostic
tool. A model like this should never be used to make real clinical
decisions on its own — any deployment in an actual healthcare setting
would need validation against a much larger, clinically-curated
dataset, regulatory review, and oversight from radiologists.

## File overview

| File | Purpose |
|---|---|
| `data_utils.py` | Dataset loading, optional raw→split preparation |
| `model.py` | CNN architectures (custom + transfer learning) |
| `train.py` | Training loop, checkpointing, class balancing |
| `evaluate.py` | Confusion matrix, ROC curve, classification report |
| `predict.py` | Single-image inference with confidence % |
| `requirements.txt` | Python dependencies |
