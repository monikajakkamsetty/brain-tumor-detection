from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import uuid

app = Flask(__name__)

MODEL_PATH = "brain_tumor_model.keras"

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

print("Model input shape:", model.input_shape)


# Upload folder
UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    confidence = None
    probability = None
    error = None
    image_url = None

    if request.method == "POST":

        print("\n-----------------------------")
        print("New image received")
        print("-----------------------------")

        if "image" not in request.files:

            error = "Please select an MRI image."

            return render_template(
                "index.html",
                error=error
            )

        file = request.files["image"]

        if file.filename == "":

            error = "Please select an MRI image."

            return render_template(
                "index.html",
                error=error
            )

        try:

            extension = os.path.splitext(file.filename)[1]

            filename = str(uuid.uuid4()) + extension

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            file.save(filepath)

            print("Image saved:", filepath)

            # Create URL for displaying image
            image_url = url_for(
                "static",
                filename="uploads/" + filename
            )

            # Prediction
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
        error=error,
        image_url=image_url
    )