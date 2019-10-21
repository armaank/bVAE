"""
model.py

contains network architecture for beta vae, as described in the appendix of [2]

"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init


def reparam(mu, logvar):
    """reparametization trick


    """

    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())

    return mu + std * eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class betaVAE(nn.Module):
    """betaVAE

    betaVAE class
    defines network architecture and initilization states

    """

    def __init__(self, z_dim=10, nchan=1):
        super(betaVAE, self).__init__()
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(nchan, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            View((-1, 32 * 4 * 4)),
            nn.Linear(32 * 4 * 4, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, z_dim * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),
            View((-1, 256, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nchan, 4, 2, 1),
        )

        self.weight_init()

    def forward(self, x):
        dist = self.encode(x)
        mu = dist[:, : self.z_dim]
        logvar = dist[:, self.z_dim :]
        z = reparam(mu, logvar)
        x_recon = self.decode(z)

        return x_recon, mu, logvar

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


if __name__ == "__main__":
    pass
