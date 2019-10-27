"""
main.py

defaults as is are set for 3d chairs

"""
import argparse

import numpy as np
import torch

from train import Trainer

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(args):
    """main

    calls Trainer to train an instance of a beta VAE and generate outputs
    inputs are the passed in cli args
    
    """

    net = Trainer(args)

    if args.train:
        net.train()
    else:
        pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="betaVAE")

    parser.add_argument("--train", default=True, type=bool, help="train or eval")
    parser.add_argument("--rand_seed", default=1, type=int, help="random seed")
    parser.add_argument("--cuda", default=True, type=bool, help="enable cuda")
    parser.add_argument("--n_iter", default=1e6, type=float, help="num of grad. steps")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--z_dim", default=7, type=int, help="dim of the latent space")
    parser.add_argument("--beta", default=5, type=float, help="beta parameter from [2]")
    parser.add_argument("--lr", default=1e-4, type=float, help="ADAM learning rate")
    parser.add_argument("--beta1", default=0.9, type=float, help="ADAM beta1")
    parser.add_argument("--beta2", default=0.999, type=float, help="ADAM beta2")
    parser.add_argument("--data_dir", default="data", type=str, help="data directory")
    parser.add_argument("--dataset", default="3dchairs", type=str, help="dataset name")
    parser.add_argument("--image_size", default=64, type=int, help="image size")
    parser.add_argument("--output_dir", default="outputs", type=str, help="output dir")
    parser.add_argument("--n_workers", default=2, type=int, help="dataloader n_workers")
    parser.add_argument(
        "--data_out",
        default="3dchairs_run1",
        type=str,
        help="folder to hold output images",
    )
    parser.add_argument(
        "--save_step",
        default=100,
        type=int,
        help="num of grad steps before checkpoints are saved",
    )

    parser.add_argument(
        "--ckpt_dir", default="checkpoints", type=str, help="checkpoint dir"
    )
    parser.add_argument(
        "--ckpt_name",
        default="last",
        type=str,
        help="load previous checkpoint. insert checkpoint filename",
    )

    args = parser.parse_args()

    main(args)
