import torch
import torch.nn as nn


class NCA(nn.Module):
    def __init__(self, n_channels=16, hidden_channels=128, filter="sobel", fire_rate=0.5, device=None):
        """
        Neural Cellular Automata Model.
        This is a PyTorch implementation of the model described in the paper https://distill.pub/2020/growing-ca/.
        We also used this implementation as a reference: https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/automata.

        Args:
            n_channels (int): number of channels
            hidden_channels (int): hidden channels
            filter (str): filter used for perception step
            fire_rate (float): waiting a random time between updates
            device (torch.device): the device we use for computation
       
        Returns:
            None
        """
        
        super(NCA, self).__init__()

        self.fire_rate = 0.5
        self.n_channels = n_channels
        self.device = device or torch.device("cpu")

        # perceive
        if filter == "sobel":
            filter_ = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            scalar = 8.0
        elif filter == "scharr":
            filter_ = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
            scalar = 16.0
        elif filter == "gaussian":
            filter_ = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
            scalar = 16.0
        elif filter == "laplacian":
            filter_ = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            scalar = 8.0
        elif filter == "mean":
            filter_ = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
            scalar = 9.0
        else:
            raise ValueError(f"Unknown filter: {filter}")

        filter_x = filter_ / scalar
        filter_y = filter_.t() / scalar

        identity = torch.tensor([[0, 0, 0],[0, 1, 0],[0, 0, 0]] ,dtype=torch.float32)
        kernel = torch.stack([identity, filter_x, filter_y], dim=0)
        kernel = kernel.repeat((n_channels, 1, 1))[:, None, ...]
        self.kernel = kernel.to(self.device)  

        # update
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
        """
        Perceive information from neighboring cells.

        Args:
            x (torch.Tensor): current grid of shape (n_samples, n_channels, grid_size, grid_size)

        Returns:
            (torch.Tensor): perceived grid of shape (n_samples, 3 * n_channels, grid_size, grid_size)
        """

        return nn.functional.conv2d(x, self.kernel, padding=1, groups=self.n_channels)


    def update(self, x):
        """
        Update cell grid.

        Args:
            x (torch.Tensor): current grid of shape (n_samples, n_channels, grid_size, grid_size)

        Returns:
            (torch.Tensor): updated grid of shape (n_samples, n_channels, grid_size, grid_size)
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
        """
        Check which cells are alive.

        Args:
            x (torch.Tensor): current grid of shape (n_samples, n_channels, grid_size, grid_size)
        
        Returns:
            (torch.Tensor): tensor with boolean values of shape (n_samples, 1, grid_size, grid_size)
        """

        return (nn.functional.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1)


    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): current grid of shape (n_samples, n_channels, grid_size, grid_size)

        Returns:
            (torch.Tensor): updated grid of shape (n_samples, n_channels, grid_size, grid_size)
        """

        return self.update(x)
