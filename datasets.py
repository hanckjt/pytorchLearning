import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from viewdatasets import viewDatasets

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
training_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# train_features, train_labels = next(iter(training_dataloader))
# img = train_features[0].squeeze()
# label = training_data.classes[train_labels[0]]
# plt.title(label)
# plt.imshow(img, cmap="gray")
# plt.show()

viewDatasets(training_data, 5, 5)