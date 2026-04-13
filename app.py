from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os   # ✅ add this

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("garbage_classifier.h5")

# Match your training labels
classes = ["Biodegradable", "Non-biodegradable", "Recyclable"]

def preprocess(image):
    image = image.convert("RGB")
    image = image.resize((160, 160))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def home():
    return "API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        image = Image.open(file)

        processed = preprocess(image)
        prediction = model.predict(processed)

        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return jsonify({
            "class": classes[class_index],
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ IMPORTANT FIX FOR RENDER
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
