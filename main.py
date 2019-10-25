"""
main.py

todo, enable cuda by default, look into visdom? 

defaults as is are set for 3d chairs

params for 3d chairs: lr=1e-4, beta1=.9, beta2=.999, batch_size=64, z_dim=16, max_iter=1e6, beta=5
params for celebA: lr=1e-4, beta1=.9, beta2=.999, batch_size=64, z_dim=32, max_iter=1e6, beta=250

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

    parser.add_argument("--train", default=True, type=bool, help="train or eval")
    parser.add_argument("--seed", default=1, type=int, help="random seed")
    parser.add_argument("--cuda", default=True, type=bool, help="enable cuda")
    parser.add_argument(
        "--max_iter", default=1e6, type=float, help="maximum number of training steps"
    )
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")

    parser.add_argument(
        "--z_dim", default=16, type=int, help="dimension of the latent space"
    )
    parser.add_argument("--beta", default=5, type=float, help="beta parameter from [2]")
    parser.add_argument("--lr", default=1e-4, type=float, help="ADAM learning rate")
    parser.add_argument("--beta1", default=0.9, type=float, help="ADAM beta1")
    parser.add_argument("--beta2", default=0.999, type=float, help="ADAM beta2")
    parser.add_argument(
        "--data_dir", default="data", type=str, help="dataset directory"
    )
    parser.add_argument("--dataset", default="3dchairs", type=str, help="dataset name")
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
