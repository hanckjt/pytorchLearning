import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

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

train_features, train_labels = next(iter(training_dataloader))
img = train_features[0].squeeze()
label = training_data.classes[train_labels[0]]
plt.title(label)
plt.imshow(img, cmap="gray")
plt.show()

# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label_idx = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(training_data.classes[label_idx])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()