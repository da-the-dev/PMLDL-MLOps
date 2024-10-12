import os
import traceback
import numpy as np
import torch
from torch import nn
from PIL import Image
from flask import Flask, jsonify, request

from src.datasets.transforms import transform
from src.datasets.labels import course_classes
from src.util.ml import load_model, device

MODEL = "effnet-b2_epoch_10.pkl"
if os.getenv("PROD"):
    MODEL = "model.pkl"

app = Flask(__name__)

model = None


@app.get("/")
def hello():
    return "Hello world!", 200


@app.post("/classify")
def classify():
    # Check if an image is part of the request
    if "image" not in request.files:
        return "No image part", 400

    file = request.files["image"]

    # If no file is selected
    if file.filename == "":
        return "No selected file", 400

    try:
        image = Image.open(file.stream)
        image = image.convert("RGB")
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.to(device)

        with torch.no_grad():
            outputs = model(image)

            sm = torch.softmax(outputs, 1)
            
            index = sm.argmax(1).item()

            return jsonify({"class": course_classes[index]}), 200

    except Exception as e:
        traceback.print_exc()

        return str(e), 500


if __name__ == "__main__":
    model = load_model(MODEL)

    model.to(device)
    model.eval()

    if os.getenv("PROD"):
        from waitress import serve
        serve(app, host="0.0.0.0", port=8000)
    else:
        app.run(debug=True)
