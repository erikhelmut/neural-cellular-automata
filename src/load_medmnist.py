from model import CAModel

# load medmnist data
import medmnist
from medmnist import INFO, Evaluator

import dataset_without_pytorch
from dataset_without_pytorch import get_loader

import matplotlib.pyplot as plt


def load_mnist_data():
    data_flag = "chestmnist"
    # data_flag = "breastmnist"
    download = True

    NUM_EPOCHS = 3
    BATCH_SIZE = 128
    lr = 0.001

    info = INFO[data_flag]
    task = info["task"]
    n_channels = info["n_channels"]
    n_classes = len(info["label"])

    DataClass = getattr(dataset_without_pytorch, info["python_class"])

    # load the data
    train_dataset = DataClass(split="train", download=download)

    # encapsulate data into dataloader form
    train_loader = get_loader(dataset=train_dataset, batch_size=BATCH_SIZE)

    return train_dataset


if __name__ == "__main__":
    train_dataset = load_mnist_data()

    px = 1/plt.rcParams["figure.dpi"]  # pixel in inches
    fig = plt.figure(frameon=False)
    fig.set_size_inches(28*px, 28*px)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(train_dataset.montage(length=1), cmap="viridis")
    fig.savefig("chest.png")