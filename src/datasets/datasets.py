import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.util.unpickle import unpickle


class CIFAR(Dataset):
    def __init__(self, path, course_labels=True, transform=None):
        self.data_path = path
        self.data_dict = unpickle(self.data_path)
        self.data = self.data_dict[b"data"]

        self.label = (
            np.array(self.data_dict[b"coarse_labels"])
            if course_labels
            else np.array(self.data_dict[b"labels"])
        )
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        img = img.reshape(3, 32, 32)
        img = img.transpose(2, 1, 0)
        if self.transform:
            img = self.transform(Image.fromarray(img))
        return img, self.label[index]

    def __len__(self):
        return len(self.data)
