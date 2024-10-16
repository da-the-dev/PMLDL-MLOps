import os
import torch
import pickle
from torch import nn
from tqdm import tqdm
from src.datasets.transforms import transform

device = "cuda" if torch.cuda.is_available() else "cpu"


def inference(model, image, labels, transform=transform):
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = nn.Softmax(dim=1)(outputs)
        class_id = torch.argmax(probabilities, dim=1)

        return labels[class_id.item()]


def train(
    epochs,
    model,
    loss_fn,
    optimizer,
    train_dl,
    test_dl,
    checkpoint_path=None,
    model_name=None,
):
    """General-purpose model training function"""
    model.to(device)

    for epoch in range(epochs):
        # Training phase
        model.train(True)
        running_loss = 0.0
        train_total = 0
        train_correct = 0

        # Training loop with progress bar
        for i, data in enumerate(
            tqdm(train_dl, desc=f"Epoch {epoch + 1}/{epochs} - Training")
        ):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Accuracy tracking during training
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_dl)
        train_accuracy = 100 * train_correct / train_total
        print(
            f"Epoch {epoch+1}/{epoch} - Avg Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%"
        )

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_total = 0
        val_correct = 0

        with torch.no_grad():
            for i, data in enumerate(
                tqdm(test_dl, desc=f"Epoch {epoch + 1}/{epochs} - Validation")
            ):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                val_running_loss += loss.item()

                # Accuracy tracking during validation
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_running_loss / len(test_dl)
        val_accuracy = 100 * val_correct / val_total
        print(
            f"Epoch {epoch+1}/{epoch} - Avg Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%\n"
        )

        # Checkpointing
        if checkpoint_path and model_name:
            save_model(
                model,
                os.path.join(checkpoint_path, f"{model_name}-epoch-{epoch+1}.pkl"),
            )


def save_model(model, path):
    with open(path, "wb+") as f:
        f.write(pickle.dumps(model))


def load_model(name):
    with open(os.path.join(os.getcwd(), "models", name), "rb") as f:
        return pickle.loads(f.read())
