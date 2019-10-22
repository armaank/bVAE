"""
Main.py


settings: ADAM optimizer wiht lr of 5e-4, beta1 .9, beta2 .999
batch_size=64
z_dim =  16
beta = 4 (according to appendix, but 250 according to fig)
max_iter = 1e6

hyper params:
adam lr, adam beta1, adam beta2, batch_size, z_dim, beta, niter

todo, enable cuda by default, look into visdom? 

"""
import argparse

import numpy as np
import torch

from train import trainer

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(args):

    betavae_net = trainer(args)

    if args.train == 1:
        betavae_net.train()
    else:
        pass
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="betaVAE")

    parser.add_argument(
        "--train", default=1, type=int, help="train or eval latent space"
    )  ## bool value for training
    parser.add_argument("--seed", default=1, type=int, help="random seed")
    parser.add_argument(
        "--max_iter", default=1e6, type=float, help="maximum training iteration"
    )
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")

    parser.add_argument(
        "--z_dim", default=10, type=int, help="dimension of the representation z"
    )
    parser.add_argument(
        "--beta",
        default=250,
        type=float,
        help="beta parameter for KL-term in original beta-VAE",
    )
    parser.add_argument("--lr", default=5e-4, type=float, help="learning rate")
    parser.add_argument("--beta1", default=0.9, type=float, help="Adam optimizer beta1")
    parser.add_argument(
        "--beta2", default=0.999, type=float, help="Adam optimizer beta2"
    )
    parser.add_argument(
        "--dset_dir", default="data", type=str, help="dataset directory"
    )
    parser.add_argument("--dataset", default="CelebA", type=str, help="dataset name")
    parser.add_argument("--image_size", default=64, type=int, help="image size")
    parser.add_argument(
        "--output_dir", default="outputs", type=str, help="output directory"
    )
    parser.add_argument(
        "--save_step",
        default=10000,
        type=int,
        help="number of iterations after which a checkpoint is saved",
    )

    parser.add_argument(
        "--ckpt_dir", default="checkpoints", type=str, help="checkpoint directory"
    )
    parser.add_argument(
        "--ckpt_name",
        default="last",
        type=str,
        help="load previous checkpoint. insert checkpoint filename",
    )

    args = parser.parse_args()

    main(args)
