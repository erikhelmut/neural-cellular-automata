import torch
import torch.nn as nn


# from: https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/automata
# and https://distill.pub/2020/growing-ca/
class CAModel(nn.Module):
    """Cellular automata model.

    n_channels : number of channels
    hidden_channels : hidden channels
    fire_rate : waiting a random time between updates
    device : the device we use for computation
    """

    def __init__(self, n_channels=16, hidden_channels=128, fire_rate=0.5, device=None):
        super().__init__()

        self.fire_rate = 0.5
        self.n_channels = n_channels
        self.device = device or torch.device("cpu")

        # Perceive step
        sobel_filter_ = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        scalar = 8.0

        sobel_filter_x = sobel_filter_ / scalar
        sobel_filter_y = sobel_filter_.t() / scalar
        identity_filter = torch.tensor(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=torch.float32,
        )
        filters = torch.stack(
            [identity_filter, sobel_filter_x, sobel_filter_y]
        )  # (3, 3, 3)
        filters = filters.repeat((n_channels, 1, 1))  # (3 * n_channels, 3, 3)
        self.filters = filters[:, None, ...].to(
            self.device
        )  # (3 * n_channels, 1, 3, 3)

        # Update step
        self.update_module = nn.Sequential(
            nn.Conv2d(
                3 * n_channels,
                hidden_channels,
                kernel_size=1,  # (1, 1)
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_channels,
                n_channels,
                kernel_size=1,
                bias=False,
            ),
        )

        with torch.no_grad():
            self.update_module[2].weight.zero_()

        self.to(self.device)

    def perceive(self, x):
        """Perceive information from neighboring cells

        x : (n_samples, n_channels, grid_size, grid_size) - current grid

        Returns
        (n_samples, 3 * n_channels, grid_size, grid_size) - perceived grid
        """
        return nn.functional.conv2d(x, self.filters, padding=1, groups=self.n_channels)

    def update(self, x):
        """Update cell grid

        x : (n_samples, 3 * n_channels, grid_size, grid_size) - current grid

        Returns
        (n_samples, n_channels, grid_size, grid_size) - updated grid
        """
        # get living cells
        pre_life_mask = self.get_alive(x)

        # perceive step
        y = self.perceive(x)
        # update step
        dx = self.update_module(y)
        # stochastic update
        device = dx.device
        mask = (torch.rand(x[:, :1, :, :].shape) <= self.fire_rate).to(device, torch.float32)
        dx = dx * mask
        # add updated value
        new_x = x + dx

        # check which cells are alive before and after
        post_life_mask = self.get_alive(new_x)
        life_mask = (pre_life_mask & post_life_mask).to(torch.float32)
        return new_x * life_mask

    @staticmethod
    def get_alive(x):
        """Check which cells are alive

        x : (n_samples, n_channels, grid_size, grid_size) - current grid

        Returns
        (n_samples, 1, grid_size, grid_size) - tensor with boolean values
        """
        return (nn.functional.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1)

    def forward(self, x):
        """Forward pass

        x : (n_samples, n_channels, grid_size, grid_size) - current grid

        Returns
        (n_sample, n_channels, grid_size, grid_size) - updated grid
        """
        return self.update(x)
