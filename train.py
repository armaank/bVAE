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

# parser.add_argument("--train", default=True, type=bool, help="train or eval")
#     parser.add_argument("--rand_seed", default=1, type=int, help="random seed")
#     parser.add_argument("--cuda", default=True, type=bool, help="enable cuda")
#     parser.add_argument("--n_iter", default=1e6, type=float, help="num of grad. steps")
#     parser.add_argument("--batch_size", default=64, type=int, help="batch size")
#     parser.add_argument("--z_dim", default=16, type=int, help="dim of the latent space")
#     parser.add_argument("--beta", default=5, type=float, help="beta parameter from [2]")
#     parser.add_argument("--lr", default=1e-4, type=float, help="ADAM learning rate")
#     parser.add_argument("--beta1", default=0.9, type=float, help="ADAM beta1")
#     parser.add_argument("--beta2", default=0.999, type=float, help="ADAM beta2")
#     parser.add_argument("--data_dir", default="data", type=str, help="data directory")
#     parser.add_argument("--dataset", default="3dchairs", type=str, help="dataset name")
#     parser.add_argument("--image_size", default=64, type=int, help="image size")
#     parser.add_argument("--output_dir", default="outputs", type=str, help="output dir")

#     parser.add_argument(
#         "--save_steps",
#         default=100,
#         type=int,
#         help="num of grad steps before checkpoints are saved",
#     )

#     parser.add_argument(
#         "--ckpt_dir", default="checkpoints", type=str, help="checkpoint dir"
#     )
#     parser.add_argument(
#         "--ckpt_name",
#         default="last",
#         type=str,
#         help="load previous checkpoint. insert checkpoint filename",
#     )


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
