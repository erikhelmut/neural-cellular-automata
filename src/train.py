import argparse
import yaml
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from model import NCA
from helper import * 


def setup_device(device=None):
    """
    Set up device for training. If no device is specified, use cuda if available,
        otherwise use mps or cpu.

    Args:
        device (str): device to use for training (defaults to None)

    Returns:
        device (torch.device): device to use for training
    """

    if device is not None:
        device = torch.device(device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def plot_loss(losses):
    """
    Plot the loss.

    Args:
        losses (list): list of losses during training

    Returns:
        None
    """

    plt.plot(losses)
    plt.title("L2 Loss during training")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()


def train(config):

    # set up device
    device = setup_device(config["device"])
    print(f"Using device: {device}")

    # load target image, pad it and repeat it batch_size times
    target = load_image(config["target_path"], config["img_size"])
    target = pad_image(target, config["padding"])
    target = target.to(device)
    target_batch = target.repeat(config["batch_size"], 1, 1, 1)

    # initialize model and optimizer
    model = NCA(n_channels=config["n_channels"], filter=config["filter"], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # initialize pool with seed cell state
    seed = make_seed(config["img_size"], config["n_channels"])
    seed = pad_image(seed, config["padding"])
    seed = seed.to(device)
    pool = seed.clone().repeat(config["pool_size"], 1, 1, 1)

    losses = []

    # start training loop
    for iter in tqdm(range(config["iterations"])):
        
        # randomly select batch_size cell states from pool
        batch_idxs = np.random.choice(
            config["pool_size"], config["batch_size"], replace=False
        ).tolist()

        # select batch_size cell states from pool
        cs = pool[batch_idxs]

        # run model for random number of iterations 
        for i in range(np.random.randint(64, 96)):
            cs = model(cs)

        # calculate loss for each image in batch
        loss_batch, loss = L2(target_batch, cs)
        losses.append(loss.item())

        # backpropagate loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # index of cell state with highest loss in batch
        argmax_batch = loss_batch.argmax().item()
        # index of cell state with highest loss in pool
        argmax_pool = batch_idxs[argmax_batch]
        # indices of cell states in batch that are not the cell state with highest loss
        remaining_batch = [i for i in range(config["batch_size"]) if i != argmax_batch]
        # indices of cell states in pool that are not the cell state with highest loss
        remaining_pool = [i for i in batch_idxs if i != argmax_pool]
        # replace cell state with highest loss in pool with seed image
        pool[argmax_pool] = seed.clone()
        # update cell states of selected batch with cell states from model output
        pool[remaining_pool] = cs[remaining_batch].detach()

    # save model
    torch.save(model.state_dict(), config["model_path"])

    # plot loss
    plot_loss(losses)


if __name__ == "__main__":
    
    # parse arguments
    parser = argparse.ArgumentParser(description="Train a NCA model.")
    parser.add_argument("-c", "--config", type=str, default="train_config.yaml", help="Path to config file.")
    args = parser.parse_args()

    # load config file
    config = yaml.safe_load(open(args.config, "r"))

    # train model
    train(config)
