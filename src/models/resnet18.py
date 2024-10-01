import os
import sys
from torch import nn, optim
from torchvision import models

from src.util.save_model import save_model
from src.util.args import args
from src.src.train_loop import train
from src.datasets.dataloaders import train_dl, test_dl, course_labels

import warnings

warnings.filterwarnings("ignore")

import logging

logger = logging.getLogger("resnet18")
logging.basicConfig(stream=sys.stdout, encoding="utf-8", level=logging.DEBUG)


def train_resnet18(epochs=10, lr=0.001):
    num_classes = 20 if course_labels else 100
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

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

    model = train_resnet18(
        epochs=args.e,
        lr=args.lr,
    )

    logger.info("Training finished. Saving...")

    save_model(model, os.path.join(os.getcwd(), "models", "resnet18.pkl"))

    logger.info("Model saved")
