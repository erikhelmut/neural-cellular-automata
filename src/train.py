import argparse
import pathlib

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from model import CAModel
from helper import * 


def train(model, optimizer, settings):
    pass


def L2_loss(x, y):
    pass

def main(argv=None):
    img_path = "../data/retina.png"
    batch_size = 8
    eval_frequency = 200 #500
    eval_iterations = 200 #300
    n_batches = 2000 + 1# 5000 + 1
    n_channels = 16
    padding = 0 #8
    pool_size = 1024
    size = 28

    # set up device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # load target image
    target_img = load_image(img_path, size)
    target_img = pad_image(target_img, padding)
    target_img = target_img.to(device)
    # how does this work with a batch size > 1?


    # Model and optimizer
    model = CAModel(n_channels=n_channels, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    # Pool initialization
    seed = make_seed(size, n_channels).to(device)
    seed = pad_image(seed, padding)
    pool = seed.clone().repeat(pool_size, 1, 1, 1)

    for iter in tqdm(range(n_batches)):
        break


if __name__ == "__main__":
    main()
