import os
import torch
from torch import nn
from PIL import Image
from flask import Flask, request

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
            probabilities = nn.Softmax(dim=1)(outputs)
            class_id = torch.argmax(probabilities, dim=1)

            return course_classes[class_id.item()], 200

    except Exception as e:
        return str(e), 500


if __name__ == "__main__":
    model = load_model(MODEL)

    model.to(device)
    model.eval()

    if os.getenv("PROD"):
        app.run(
            host="0.0.0.0",
            port=8000,
        )
    else:
        app.run(debug=True)
