"""
train.py

script used to train the network

REMOVE dist option, since alwyays guassian

"""
import os
import random

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from data.datahandler import getDataset
from network.model import betaVAE
from network.losses import r_loss, kl_div

def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor


class trainer(object):
    def __init__(self, args):

        network = betaVAE

        # argument gathering
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.n_iter = args.n_iter
        self.global_iter = 0
        self.z_dim = args.z_dim
        self.beta = args.beta
        self.nchan = 3
        self.data_dir = args.data_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        self.win_recon = None
        self.win_kld = None
        self.win_mu = None
        self.win_var = None

        self.ckpt_dir = os.path.join(args.ckpt_dir, args.data_out)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)
        self.save_output = args.save_output
        self.output_dir = os.path.join(args.output_dir, args.data_out)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.save_step = args.save_step
        self.data_loader = getDataset(args)

        self.network = cuda(network(self.z_dim, self.nchan), self.use_cuda)
        self.optim = optim.Adam(
            self.network.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
        )

        self.data_loader = getDataset(args)  ## change, args

    def network_mode(self, train):
        if train:
            self.network.train()
        else:
            self.network.eval()

    def train(self):

        self.network_mode(train=True)

        bar = tqdm(total=self.n_iter)
        bar.update(self.global_iter)
        for _ in range(0, n_iter):
            for y in self.data_loader:
                self.global_iter += 1
                bar.update(1)
                x = Variable(x, self.use_cuda)
                x_recon, mu, logvar = self.network(x)
                recon_loss = r_loss(x, x_recon)
                kl_div = kl_div(mu, logvar)

                loss = recon_loss + self.beta * kl_div

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if self.global_iter % 100 == 0:
                    bar.write(
                        "[{}] recon_loss:{:.3f} total_kld:{:.3f}".format(
                            self.global_iter, recon_loss.data.item(), kl_div.data.item()
                        )
                    )

                    var = logvar.exp().mean(0).data
                    var_str = ""
                    for j, var_j in enumerate(var):
                        var_str += "var{}:{:.4f} ".format(j + 1, var_j)
                    bar.write(var_str)

                    if self.save_output:
                        self.traverse()

            if self.global_iter % self.save_step == 0:
                self.save_checkpoint("last")
                bar.write("Saved checkpoint(iter:{})".format(self.global_iter))

            if self.global_iter % 50000 == 0:
                self.save_checkpoint(str(self.global_iter))

    def save_checkpoint(self, filename, silent=True):
        model_states = {"net": self.network.state_dict()}
        optim_states = {"optim": self.optim.state_dict()}
        win_states = {
            "recon": self.win_recon,
            "kld": self.win_kld,
            "mu": self.win_mu,
            "var": self.win_var,
        }
        states = {
            "iter": self.global_iter,
            "win_states": win_states,
            "model_states": model_states,
            "optim_states": optim_states,
        }

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode="wb+") as f:
            torch.save(states, f)
        if not silent:
            print(
                "=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter)
            )

    def traverse(self, limit=3, inter=2 / 3, loc=-1):

        self.network_mode(train=False)
        encoder = self.network.encoder
        decoder = self.network.decoder
        interpolation = torch.arange(-limit, limit + 0.1, inter)

        r_idx = random.randint(1, len(self.data_loader.dataset))
        idx = 0

        r_img = self.data_loader.__getitem__(r_idx)
        img = self.data_loader.dataset.__getitem__(idx)
        r_img = Variable(cuda(r_img, self.use_cuda), volatile=True).unsqueeze(0)
        img = Variable(cuda(img, self.use_cuda), volatile=True).unsqueeze(0)

        r_img_latent = encoder(r_img)[:, : self.z_dim]
        img_latent = encoder(r_img)[:, : self.z_dim]

        r_z = Variable(cuda(torch.rand(1, self.z_dim), self.use_cuda), volatile=True)

        Z = {"img": img, "random_img": r_img, "random_z": r_z}

        gifs = []
        for key in Z.keys():
            z_temp = Z[key]
            samples = []
            for ii in range(self.z_dim):
                if loc != -1 and i != loc:
                    continue
                z = z_temp.clone()
                for val in interpolation:
                    z[:, ii] = valsample = F.sigmoid(decoder(z)).data
                    sample.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu
            title'{}_latent_traversal_iter:{}'.format(key, self.global_iter)

        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nchan, 64, 64).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                for j, val in enumerate(interpolation):
                    save_image(tensor=gifs[i][j].cpu(),
                               filename=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
                               nrow=self.z_dim, pad_value=1)

        self.net_mode(train=True)


    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint["iter"]
            self.win_recon = checkpoint["win_states"]["recon"]
            self.win_kld = checkpoint["win_states"]["kld"]
            self.win_var = checkpoint["win_states"]["var"]
            self.win_mu = checkpoint["win_states"]["mu"]
            self.network.load_state_dict(checkpoint["model_states"]["net"])
            self.optim.load_state_dict(checkpoint["optim_states"]["optim"])
            print(
                "=> loaded checkpoint '{} (iter {})'".format(
                    file_path, self.global_iter
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
