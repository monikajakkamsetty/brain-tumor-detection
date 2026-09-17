import os
import uuid
 
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
import numpy as np
 
# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "brain_tumor_model.keras"
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB
 
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
 
# -----------------------------
# Load model
# -----------------------------
model = None
model_load_error = None
 
try:
    from tensorflow.keras.models import load_model
 
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'")
 
    print("Loading model...", flush=True)
    model = load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully!", flush=True)
    print("Model input shape:", model.input_shape, flush=True)
 
except Exception as e:
    model_load_error = str(e)
    print("ERROR loading model:", model_load_error, flush=True)
 
 
# -----------------------------
# Helpers
# -----------------------------
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )
 
 
def predict_image(image_path):
    """
    Runs the model on the given image and returns:
      result       -> "Tumor Detected" / "No Tumor Detected"
      confidence   -> confidence in the predicted class, as a percentage
      probability  -> raw probability that the image shows a tumor
    Handles both a single sigmoid output (shape [..., 1]) and a
    two-class softmax output (shape [..., 2]).
    """
    input_shape = model.input_shape
    height = input_shape[1]
    width = input_shape[2]
    channels = input_shape[3] if len(input_shape) > 3 else 3
 
    mode = "L" if channels == 1 else "RGB"
 
    image = Image.open(image_path).convert(mode)
    image = image.resize((width, height))
 
    image_array = np.array(image).astype("float32") / 255.0
 
    if channels == 1 and image_array.ndim == 2:
        image_array = np.expand_dims(image_array, axis=-1)
 
    image_array = np.expand_dims(image_array, axis=0)
 
    prediction = model.predict(image_array, verbose=0)
    print("Raw model output:", prediction, flush=True)
 
    output = prediction[0]
 
    if output.shape[-1] == 1:
        # Single sigmoid unit: probability of "tumor" class
        probability = float(output[0])
    elif output.shape[-1] == 2:
        # Two-class softmax: assume index 1 == "tumor"
        probability = float(output[1])
    else:
        raise ValueError(
            f"Unexpected model output shape: {output.shape}. "
            "Expected a single sigmoid value or a 2-class softmax."
        )
 
    if probability >= 0.5:
        result = "Tumor Detected"
        confidence = probability * 100
    else:
        result = "No Tumor Detected"
        confidence = (1 - probability) * 100
 
    return result, confidence, probability
 
 
# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    probability = None
    error = None
    image_url = None
 
    if request.method == "POST":
        print("\n----------------------------- New image received", flush=True)
 
        if model is None:
            error = f"Model failed to load: {model_load_error}"
            return render_template("index.html", error=error)
 
        if "image" not in request.files or request.files["image"].filename == "":
            error = "Please select an MRI image."
            return render_template("index.html", error=error)
 
        file = request.files["image"]
 
        if not allowed_file(file.filename):
            error = "Unsupported file type. Please upload a JPG, JPEG, or PNG image."
            return render_template("index.html", error=error)
 
        try:
            extension = secure_filename(file.filename).rsplit(".", 1)[1].lower()
            filename = f"{uuid.uuid4()}.{extension}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
 
            file.save(filepath)
            print("Image saved:", filepath, flush=True)
 
            image_url = url_for("static", filename=f"uploads/{filename}")
 
            result, confidence, probability = predict_image(filepath)
 
            print("RESULT:", result, flush=True)
            print("CONFIDENCE:", confidence, flush=True)
            print("PROBABILITY:", probability, flush=True)
 
        except UnidentifiedImageError:
            error = "That file doesn't look like a valid image. Please try another."
        except Exception as e:
            print("ERROR:", repr(e), flush=True)
            error = f"Prediction error: {e}"
 
    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        probability=probability,
        error=error,
        image_url=image_url,
    )
 
 
@app.errorhandler(413)
def too_large(e):
    return render_template(
        "index.html", error="File too large. Please upload an image under 8 MB."
    ), 413
 
 
# -----------------------------
# Local development
# -----------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )
 