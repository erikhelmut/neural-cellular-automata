import torch
import torch.nn as nn

class NCA(nn.Module):

    def __init__(self, n_channels=16, hidden_channels=128, fire_rate=0.5, device=None):
        super(NCA, self).__init__()

        self.n_channels = n_channels
        self.hidden_channels = hidden_channels
        self.fire_rate = fire_rate

        # Set the device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ### perception ###
        sobel_filter = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)

        sobel_filter_x = sobel_filter / torch.sum(torch.abs(sobel_filter))
        sobel_filter_y = sobel_filter_x.t()

        identity_filter = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)

        filters = torch.stack([identity_filter, sobel_filter_x, sobel_filter_y])
        filters = filters.repeat((n_channels, 1, 1))
        self.filters = filters[:, None, ...].to(self.device)

        ### update step ###
        self.conv1 = nn.Conv2d(3 * n_channels, hidden_channels, kernel_size=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_channels, n_channels, kernel_size=1, padding=1, bias=False)

    def perceive(self, x):
        return nn.functional.conv2d(x, self.filters, padding=1)

    def update(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out

    def stochastic_update(self, x, fire_rate):
        device = x.device

        mask = (torch.rand(x[:, :1, :, :].shape) <= fire_rate).to(device, torch.float32)
        return x * mask

    def alive_masking(self, x):
        return (
                nn.functional.max_pool2d(
                    x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1
                )
                > 0.1
        )

    def forward(self, x):
        pre_life_mask = self.alive_masking(x)

        fixedFilter = self.perceive(x)
        update = self.update(fixedFilter)
        stochasticUpdate = self.stochastic_update(update, self.fire_rate)

        # step
        x = x + stochasticUpdate

        # alive masking
        post_life_mask = self.alive_masking(x)
        out = (pre_life_mask & post_life_mask).to(torch.float32)

        return x * out


if __name__ == "__main__":
    nca = NCA()
