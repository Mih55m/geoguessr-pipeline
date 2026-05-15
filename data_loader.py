#data_loader.py
from torchvision import datasets
from collections import Counter

DATA_DIR = "./dataset"

def get_class_names():
    dataset = datasets.ImageFolder(root=DATA_DIR)
    return dataset.classes

def get_class_counts():
    dataset = datasets.ImageFolder(root=DATA_DIR)
    class_counts = Counter(dataset.targets)
    return class_counts
