import torch
import matplotlib.pyplot as plt


def viewDatasets(ds, cols=3, rows=3, cmap="gray"):
    figure = plt.figure(figsize=(8, 8))
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(ds), size=(1,)).item()
        img, label_idx = ds[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(ds.classes[label_idx])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap=cmap)
    plt.show()