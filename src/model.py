import torch
import torch.nn as nn


# from: https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/automata
# and https://distill.pub/2020/growing-ca/
class CAModel(nn.Module):
    """Cellular automata model.

    n_channels : number of channels
    hidden_channels : hidden channels
    wait_time : waiting a random time between updates
    device : the device we use for computation
    """

    def __init__(self, n_channels=16, hidden_channels=128, wait_time=0.5, device=None):
        super().__init__()
        # initialize attributes
        self.wait_time = wait_time
        self.n_channels = n_channels

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        # initialize values for perception step
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = sobel_x.t()
        identity_filter = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0], ], dtype=torch.float32, )
        # stack filters
        filters = torch.stack(
            [identity_filter, sobel_x, sobel_y]
        )
        # because we have 3 * n_channels in the first update_module
        filters = filters.repeat((n_channels, 1, 1))
        self.filters = filters[:, None, :, :].to(self.device)

        # dense-128, relu, dense-16
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
        mask = (torch.rand(x[:, :1, :, :].shape) <= self.wait_time).to(self.device, torch.float32)
        x = x * mask
        # add updated value
        x = x + dx

        # check which cells are alive before and after
        post_life_mask = self.get_alive(x)
        life_mask = (pre_life_mask & post_life_mask).to(torch.float32)
        x = x * life_mask
        return x

    @staticmethod
    def get_alive(x):
        """Check which cells are alive

        x : (n_samples, n_channels, grid_size, grid_size) - current grid

        Returns
        (n_samples, 1, grid_size, grid_size) - tensor with boolean values
        """
        return nn.functional.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def forward(self, x):
        """Forward pass

        x : (n_samples, n_channels, grid_size, grid_size) - current grid

        Returns
        (n_sample, n_channels, grid_size, grid_size) - updated grid
        """
        x = self.update(x)
        return x