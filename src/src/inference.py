import torch
from torch import nn

from src.datasets.transforms import transform
from src.util.device import device


def inference(model, image, labels, transform=transform):
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = nn.Softmax(dim=1)(outputs)
        class_id = torch.argmax(probabilities, dim=1)

        return labels[class_id.item()]
