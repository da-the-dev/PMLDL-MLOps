import os
import pickle
import torch
from torch import nn
from PIL import Image
from flask import Flask, jsonify, request

from src.datasets.transforms import transform


app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_file = open(os.path.join(os.getcwd(), "models", "effnet-b2_epoch_10.pkl"), "rb")
model = None


course_classes = [
    "aquatic mammals",
    "fish",
    "flowers",
    "food containers",
    "fruit and vegetables",
    "household electrical devices",
    "household furniture",
    "insects",
    "large carnivores",
    "large man-made outdoor things",
    "large natural outdoor scenes",
    "large omnivores and herbivores",
    "medium-sized mammals",
    "non-insect invertebrates",
    "people",
    "reptiles",
    "small mammals",
    "trees",
    "vehicles 1",
    "vehicles 2",
]


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
    model = pickle.loads(model_file.read())

    model.to(device)
    model.eval()
    app.run(debug=True)
