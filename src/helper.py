import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import medmnist
from medmnist import INFO, Evaluator
import medmnist_loader
from medmnist_loader import get_loader
import matplotlib.pyplot as plt


def load_medmnsit_data(data_flag="chestmnist"):
    """
    Load the medmnist data for the given data flag.
        Source: https://github.com/MedMNIST/MedMNIST

    Args:
        data_flag (str): data flag for the medmnist dataset (defaults to "chestmnist")

    Returns:
        train_dataset (MedMNIST): training dataset
        data_flag (str): data flag for the medmnist dataset
    """

    download = True

    NUM_EPOCHS = 3
    BATCH_SIZE = 128
    lr = 0.001

    info = INFO[data_flag]
    task = info["task"]
    n_channels = info["n_channels"]
    n_classes = len(info["label"])

    DataClass = getattr(medmnist_loader, info["python_class"])

    train_dataset = DataClass(split="train", download=download)

    train_loader = get_loader(dataset=train_dataset, batch_size=BATCH_SIZE)

    return train_dataset, data_flag


def save_medmnist_image(train_dataset, data_flag):
    """
    Save the medmnist image for the given data flag.

    Args:
        train_dataset (MedMNIST): training dataset
        data_flag (str): data flag for the medmnist dataset

    Returns:
        None
    """

    px = 1/plt.rcParams["figure.dpi"]  # pixel in inches
    fig = plt.figure(frameon=False)
    fig.set_size_inches(28*px, 28*px)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(train_dataset.montage(length=1), cmap="viridis")
    fig.savefig("../data/" + data_flag + ".png")


def load_image(path, size=28):
    """
    Load an image and convert it to a tensor.

    Args:
        path (str): path to image
        size (int, optional): max size of image (defaults to 28)

    Returns:
        img (torch.Tensor): image of shape (1, 4, size, size) where the first three
            channels are RGB and the last channel is the alpha channel
    """
    
    # load image and resize
    img = Image.open(path).resize((size, size), Image.ANTIALIAS)

    # convert to float and normalize
    img = np.float32(img) / 255.0
    
    # premultiply RGB channels by alpha channel
    img[..., :3] *= img[..., 3:]

    # convert to tensor and permute dimensions
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    return img


def rgba_to_rgb(img):
    """
    Convert an RGBA image to an RGB image.
    
    Args:
        img (torch.Tensor): image of shape (1, 4, size, size) where the first three
            channels are RGB and the last channel is the alpha channel
    
    Returns:
        img (torch.Tensor): image of shape (1, 3, size, size) where the first three
            channels are RGB
    """
    
    # separate RGB and alpha channels
    rgb = img[:, :3, ...]
    alpha = torch.clamp(img[:, 3:4, ...], 0.0, 1.0)

    # convert to RGB
    img = torch.clamp(1.0 - alpha + rgb, 0, 1)

    return img


def pad_image(img, p=0):
    """
    Pad an image with zeros.

    Args:
        img (torch.Tensor): image of shape (1, n_channels, size, size) 
        p (int, optional): number of pixels to pad image (defaults to 0)
    
    Returns:
        img (torch.Tensor): padded image of shape (1, n_channels, size + 2p, size + 2p)
    """
    
    img = nn.functional.pad(img, (p, p, p, p), mode="constant", value=0)
    
    return img


def make_seed(size, n_channels=16):
    """
    Initialize the grid with zeros, except a single seed cell in the center, 
        which will have all channels except RGB 2 set to one.

    Args:
        size (int): size of the image
        n_channels (int): number of channels. Defaults to 16 and must be greater
            than 4, because the first 3 channels are RGB and the 4th channel is
            the alpha channel
    
    Returns:
        x (torch.Tensor): initialization grid of shape (1, n_channels, size, size)
    """

    if n_channels < 4:
        raise ValueError("n_channels must be greater than 4")

    x = torch.zeros((1, n_channels, size, size), dtype=torch.float32)
    x[:, 3:, size // 2, size // 2] = 1.0

    return x


def L2(target, cs):
    """
    Calculate the L2 loss between target image and cell state.

    Args:
        target (torch.Tensor): target image of shape (batch_size, 4, size, size)
        cs (torch.Tensor): cell state

    Returns:
        loss_batch (torch.Tensor): L2 loss for each image in batch
        loss (torch.Tensor): L2 loss
    """

    # calculate loss for each image in batch but only take first 4 rgba channels
    loss_batch = ((target - cs[:, :4, ...]) ** 2).mean(dim=[1, 2, 3])

    # take mean over loss_batch
    loss = loss_batch.mean()

    return loss_batch, loss