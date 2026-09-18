import base64
import io
import logging
import os
import threading
 
# Must be set before TensorFlow is imported
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
 
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
 
# -----------------------------
# Config
# -----------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "brain_tumor_model.keras")
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB
 
# Actual image formats accepted (checked from file contents, not the filename)
ALLOWED_FORMATS = {"PNG": "png", "JPEG": "jpeg"}
 
# These MUST match how the model was trained:
RESCALE = 1.0 / 255.0     # set to 1.0 if the model has its own Rescaling layer
TUMOR_CLASS_INDEX = 1     # index of "tumor" for a 2-class softmax output
THRESHOLD = 0.5
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("brain-tumor-app")
 
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
 
 
class UserError(Exception):
    """An error whose message is safe to show to the user."""
 
 
# -----------------------------
# Model loading (background thread)
#
# The model is loaded in a background thread so gunicorn binds the port and
# answers requests immediately. Loading TensorFlow at import time blocks the
# worker, hits gunicorn's timeout, and the proxy returns 502.
# -----------------------------
_model = None
_model_error = None

_state_lock = threading.Lock()
_predict_lock = threading.Lock()
_model_ready_event = threading.Event()
_loader_pid = None


def _load_model():
    global _model, _model_error

    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'"
            )

        logger.info("Loading model...")

        from tensorflow.keras.models import load_model

        model = load_model(MODEL_PATH, compile=False)

        logger.info(
            "Model loaded. Input shape: %s",
            model.input_shape
        )

        # Warm-up prediction
        shape = [1] + [
            d or 1 for d in model.input_shape[1:]
        ]

        model(
            np.zeros(shape, dtype="float32"),
            training=False
        )

        _model = model

        logger.info("Model ready.")

    except Exception as e:
        _model_error = str(e)
        logger.exception("Failed to load model")

    finally:
        _model_ready_event.set()


def start_model_loading():
    """Start model loading once per process."""
    global _loader_pid

    with _state_lock:

        if _loader_pid == os.getpid():
            return

        _loader_pid = os.getpid()

        threading.Thread(
            target=_load_model,
            daemon=True
        ).start()


start_model_loading()
 
# -----------------------------
# Helpers
# -----------------------------
def decode_image(data):
    """Open and validate an uploaded image from raw bytes."""
    try:
        img = Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, OSError):
        raise UserError("That file doesn't look like a valid image. Please try another.")
 
    if img.format not in ALLOWED_FORMATS:
        raise UserError("Unsupported file type. Please upload a JPG, JPEG, or PNG image.")
    return img
 
 
def predict_image(img):
    """
    Returns (result, confidence_percent, tumor_probability).
    Supports a single sigmoid output or a 2-class softmax output.
    """
    model = _model
    input_shape = model.input_shape
    height, width = input_shape[1], input_shape[2]
    channels = input_shape[3] if len(input_shape) > 3 and input_shape[3] else 3
 
    img = ImageOps.exif_transpose(img)
    img = img.convert("L" if channels == 1 else "RGB")
    img = img.resize((width, height), Image.Resampling.BILINEAR)
 
    arr = np.asarray(img, dtype="float32") * RESCALE
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    batch = np.expand_dims(arr, axis=0)
 
    with _predict_lock:
        output = np.ravel(np.asarray(model(batch, training=False)))
 
    if output.size == 1:
        probability = float(output[0])
    elif output.size == 2:
        probability = float(output[TUMOR_CLASS_INDEX])
    else:
        raise ValueError(
            f"Unexpected model output size: {output.size}. "
            "Expected a single sigmoid value or a 2-class softmax."
        )
 
    if probability >= THRESHOLD:
        return "Tumor Detected", probability * 100, probability
    return "No Tumor Detected", (1 - probability) * 100, probability
 
 
# -----------------------------
# Routes
# -----------------------------
@app.route("/healthz")
def healthz():
    """Responds immediately, even while the model is still loading."""
    return jsonify(status="ok", model_ready=_model is not None), 200
 
 
@app.route("/", methods=["GET", "POST"])
def home():
    start_model_loading()
 
    if request.method == "GET":
        return render_template("index.html")
 
    # ---- POST ----
    if _model is None:

    # Wait up to 120 seconds for TensorFlow model loading
    _model_ready_event.wait(timeout=120)

    if _model_error:
        logger.error(
            "Model unavailable: %s",
            _model_error
        )

        return render_template(
            "index.html",
            error="The prediction model is unavailable. Please try again later.",
        ), 500

    if _model is None:
        return render_template(
            "index.html",
            error="The model could not be loaded. Please try again later.",
        ), 500
 
    file = request.files.get("image")
    if file is None or file.filename == "":
        return render_template("index.html", error="Please select an MRI image.")
 
    try:
        data = file.read()
        img = decode_image(data)
        image_format = ALLOWED_FORMATS[img.format]
 
        result, confidence, probability = predict_image(img)
 
        # Preview is embedded in the page; nothing is written to disk.
        image_url = f"data:image/{image_format};base64,{base64.b64encode(data).decode()}"
 
        logger.info("Prediction: %s (p=%.4f)", result, probability)
        return render_template(
            "index.html",
            result=result,
            confidence=confidence,
            probability=probability,
            image_url=image_url,
        )
 
    except UserError as e:
        return render_template("index.html", error=str(e))
    except Image.DecompressionBombError:
        return render_template("index.html", error="That image is too large to process.")
    except Exception:
        logger.exception("Prediction failed")
        return render_template(
            "index.html",
            error="Something went wrong while analysing the image. Please try another.",
        ), 500
 
 
@app.errorhandler(413)
def too_large(e):
    return render_template(
        "index.html", error="File too large. Please upload an image under 8 MB."
    ), 413
 
 
# -----------------------------
# Local development
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
 