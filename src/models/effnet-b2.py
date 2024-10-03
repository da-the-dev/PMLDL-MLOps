import os
import sys
from torch import nn, optim
from torchvision import models
from efficientnet_pytorch import EfficientNet

from src.util.ml import save_model, train
from src.util.args import args
from src.datasets.dataloaders import train_dl, test_dl, course_labels

import warnings

warnings.filterwarnings("ignore")

import logging

logger = logging.getLogger("resnet18")
logging.basicConfig(stream=sys.stdout, encoding="utf-8", level=logging.DEBUG)


def train_effnet_b2(epochs=10, lr=0.001):
    num_classes = 20 if course_labels else 100
    model = EfficientNet.from_pretrained("efficientnet-b2")
    model._fc = nn.Linear(model._fc.in_features, num_classes)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    train(
        epochs,
        model,
        loss_fn,
        optimizer,
        train_dl,
        test_dl,
    )

    return model


if __name__ == "__main__":
    args = args()

    logger.info("Model training started...")
    logger.info(f"Epochs: {args.e}")
    logger.info(f"LR: {args.lr}")

    model = train_effnet_b2(
        epochs=args.e,
        lr=args.lr,
    )

    logger.info("Training finished. Saving...")

    save_model(model, os.path.join(os.getcwd(), "models", "effnet-b2.pkl"))

    logger.info("Model saved")
