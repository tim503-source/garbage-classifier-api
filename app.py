from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("garbage_classifier.h5")

# Match your training labels
classes = ["Biodegradable", "Non-biodegradable", "Recyclable"]

def preprocess(image):
    image = image.convert("RGB")              # ✅ FIX: ensure 3 channels
    image = image.resize((160, 160))          # ✅ FIX: correct input size
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def home():
    return "API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if image is sent
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
        return jsonify({"error": str(e)}), 500   # ✅ shows real error

if __name__ == "__main__":
    app.run(debug=True)