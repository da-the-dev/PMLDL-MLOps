import os
import sys
from torch import nn, optim
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet

from src.datasets.transforms import transform
from src.datasets.datasets import CIFAR
from src.util.ml import save_model, train
from src.util.args import args

import warnings

warnings.filterwarnings("ignore")

import logging

logger = logging.getLogger("effnet-b2")
logging.basicConfig(stream=sys.stdout, encoding="utf-8", level=logging.DEBUG)


def train_effnet_b2(
    epochs=10,
    lr=0.001,
    data_path=None,
    transform=None,
    course_labels=True,
    batch_size=32,
):
    # Define train and test data
    train_dataset = CIFAR(os.path.join(data_path, "train"), course_labels, transform)
    test_dataset = CIFAR(os.path.join(data_path, "test"), course_labels, transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define model
    num_classes = 20 if course_labels else 100
    model = EfficientNet.from_pretrained("efficientnet-b2")
    model._fc = nn.Linear(model._fc.in_features, num_classes)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    train(
        epochs,
        model,
        loss_fn,
        optimizer,
        train_dataloader,
        test_dataloader,
        checkpoint_path="models",
        model_name="effnet_b2",
    )

    return model


if __name__ == "__main__":
    args = args()

    logger.info("Model training started...")

    model = train_effnet_b2(
        transform=transform,
        epochs=args.epochs,
        lr=args.lr,
        data_path=args.data_path,
        course_labels=args.course_labels,
        batch_size=args.batch_size,
    )

    logger.info("Training finished")
