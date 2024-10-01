import os
from torch.utils.data import DataLoader
from src.datasets.datasets import CIFAR
from src.datasets.transforms import transform

batch_size = 1024
course_labels = True

train_data = CIFAR(os.path.join(os.getcwd(), 'data', 'train'), course_labels, transform)
test_data = CIFAR(os.path.join(os.getcwd(), 'data', 'test'), course_labels, transform)

train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=False)