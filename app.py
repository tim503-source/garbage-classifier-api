from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import os
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

# Load TFLite model (Drastically lighter than full TensorFlow for Render Free Tier)
MODEL_PATH = "garbage_classifier.tflite"
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Match your training labels
classes = ["Biodegradable", "Non-biodegradable", "Recyclable"]

def preprocess(image):
    image = image.convert("RGB")
    image = image.resize((160, 160))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def home():
    return "API is running (TFLite Mode)"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = None
        if "image" in request.files:
            file = request.files["image"]
        elif "file" in request.files:
            file = request.files["file"]

        if not file:
            return jsonify({"error": "No image uploaded. Use 'image' or 'file' key."}), 400

        image = Image.open(file)
        processed = preprocess(image)

        # TFLite Inference
        interpreter.set_tensor(input_details[0]['index'], processed)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return jsonify({
            "class": classes[class_index],
            "label": classes[class_index],
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)