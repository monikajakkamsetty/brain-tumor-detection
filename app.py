from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import uuid

app = Flask(__name__)

# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = "brain_tumor_model.keras"

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

print("Model input shape:", model.input_shape)


# -----------------------------
# Upload folder
# -----------------------------
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# -----------------------------
# Prediction function
# -----------------------------
def predict_image(image_path):

    # Get model input size automatically
    input_shape = model.input_shape

    height = input_shape[1]
    width = input_shape[2]

    print("Using image size:", width, "x", height)

    # Open image
    image = Image.open(image_path).convert("RGB")

    # Resize to model input size
    image = image.resize((width, height))

    # Convert to numpy
    image_array = np.array(image).astype("float32")

    # Normalize
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array, verbose=0)

    print("Raw model output:", prediction)

    # Get probability
    probability = float(prediction[0][0])

    print("Tumor probability:", probability)

    # Classification
    if probability >= 0.5:

        result = "Tumor Detected"

        confidence = probability * 100

    else:

        result = "No Tumor Detected"

        confidence = (1 - probability) * 100

    return result, confidence, probability


# -----------------------------
# Home page
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    confidence = None
    probability = None
    error = None

    if request.method == "POST":

        print("\n-----------------------------")
        print("New image received")
        print("-----------------------------")

        # Check file
        if "image" not in request.files:

            error = "Please select an MRI image."

            return render_template(
                "index.html",
                error=error
            )

        file = request.files["image"]

        # Check filename
        if file.filename == "":

            error = "Please select an MRI image."

            return render_template(
                "index.html",
                error=error
            )

        try:

            # Get extension
            extension = os.path.splitext(file.filename)[1]

            # Generate unique filename
            filename = str(uuid.uuid4()) + extension

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            # Save image
            file.save(filepath)

            print("Image saved:", filepath)

            # Run prediction
            result, confidence, probability = predict_image(filepath)

            print("RESULT:", result)
            print("CONFIDENCE:", confidence)
            print("PROBABILITY:", probability)

        except Exception as e:

            print("ERROR:", e)

            error = f"Prediction error: {e}"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        probability=probability,
        error=error
    )


# -----------------------------
# Run application
# -----------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )