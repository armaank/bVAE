"""
model.py

contains network architecture for beta vae, as described in [3]

"""
import torch
import torch.nn as nn
from torch.autograd import Variable


def reparam_trick(mu, logvar):
    """reparametization trick


    """

    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())

    return mu + std * eps

class betaVAE(nn.Module):

    