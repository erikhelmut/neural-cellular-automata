import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from model import CAModel
from helper import *

# load model
model = CAModel(n_channels=16, device=torch.device("cpu"))
model.load_state_dict(torch.load("../models/chest.pt", map_location="cpu"))
model.eval()

# load target image
img_path = "../data/chest.png"
target_img = load_image(img_path, size=28)

frames = []
fig, ax = plt.subplots()

# run model
x = make_seed(28, 16)
for i in tqdm(range(2000)):
    x = model(x)
    frame = ax.imshow(rgba_to_rgb(x[:, :4].detach().cpu())[0].permute(1, 2, 0), animated=True) 
    frames.append([frame])
ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
ani.save("../pictures/chest.gif", writer="imagemagick")
plt.show()
