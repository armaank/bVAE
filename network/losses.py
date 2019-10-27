"""
losses.py

methods used to form the objective function described in [2]

"""
import os

import torch

import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image


def r_loss(x, x_recon):
    """r_loss

    """

    batch_size = x.size(0)
    x_recon = torch.sigmoid(x_recon)
    recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)

    return recon_loss


def kl_div(mu, logvar):
    """kl_div


    """

    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)

    return total_kld
