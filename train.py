"""
train.py

script used to train the network

REMOVE dist option, since alwyays guassian

"""
import os

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from data.datahandler import getDataset
from network.model import betaVAE
from network.losses import r_loss, kl_div


class trainer(object):
    def __init__(self, args):

        # argument gathering
        self.max_iter = args.max_iter
        self.global_iter = 0
        self.z_dim = args.z_dim
        self.beta = args.beta
        self.objective = args.objective

        if args.dataset.lower() == "3dchairs":
            self.nchan = 3
            self.dist = "gaussian"
        elif args.dataset.lower() == "celeba":
            self.nchan = 3
            self.dist = "gaussian"

        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size

        self.data_loader = getDataset(args)  ## change, args

    def train(self):

        pass
