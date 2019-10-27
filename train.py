"""
train.py

script used to train the network and generate outputs

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


class Trainer(object):
    """Trainer
    
    Trainer class used to instantiate a betaVAE, 
    train it and generate outputs from the latent spcae

    """

    def __init__(self, args):

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

        self.net = cuda(betaVAE(self.z_dim, self.nchan), self.use_cuda)
        self.optim = optim.Adam(
            self.net.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
        )

        self.ckpt_dir = os.path.join(args.ckpt_dir, args.data_out)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_ckpt(self.ckpt_name)

        self.output_dir = os.path.join(args.output_dir, args.data_out)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.save_step = args.save_step
        self.data_loader = getDataset(args)

    def train(self):
        """train

        trains an instance of a betaVAE

        """

        self.net.train()

        bar = tqdm(total=self.n_iter)
        bar.update(self.global_iter)
        out = False
        while not out:
            for x in self.data_loader:
                self.global_iter += 1
                bar.update(1)
                x = Variable(cuda(x, self.use_cuda))
                x_recon, mu, logvar = self.net(x)
                reconloss = r_loss(x, x_recon)
                kld = kl_div(mu, logvar)

                loss = reconloss + self.beta * kld

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if self.global_iter % self.save_step == 0:
                    bar.write(
                        "[{}] recon_loss:{:.3f} total_kld:{:.3f}".format(
                            self.global_iter, reconloss.data.item(), kld.data.item()
                        )
                    )

                    self.traverse_latent()

                if self.global_iter % self.save_step == 0:
                    self.save_ckpt("last")
                    bar.write("Saved checkpoint(iter:{})".format(self.global_iter))

                if self.global_iter % self.save_step == 0:
                    self.save_ckpt(str(self.global_iter))

                if self.global_iter >= self.n_iter:
                    out = True
                    break

        bar.write("[done training]")
        bar.close()

    def traverse_latent(self, limit=3, inter=1):
        """traverse_latent

        function to traverse latent space and generate outputs

        inputs: range over which to traverse the latent space 
                default is [-3,3], as perscribed by [2]
        outputs: output images to make Fig. 1 and Fig. 2 in [2]

        """

        encoder = self.net.encoder
        decoder = self.net.decoder
        interp = torch.arange(-limit, limit, inter)

        r_idx = random.randint(1, len(self.data_loader.dataset))
        idx = 0

        r_img = self.data_loader.dataset.__getitem__(r_idx)
        img = self.data_loader.dataset.__getitem__(idx)
        r_img = Variable(cuda(r_img, self.use_cuda), volatile=True).unsqueeze(0)
        img = Variable(cuda(img, self.use_cuda), volatile=True).unsqueeze(0)

        r_img_latent = encoder(r_img)[:, : self.z_dim]
        img_latent = encoder(r_img)[:, : self.z_dim]

        r_z = Variable(cuda(torch.rand(1, self.z_dim), self.use_cuda), volatile=True)

        Z = {"img": img_latent, "random_img": r_img_latent, "random_z": r_z}

        images = []
        for key in Z.keys():
            z_temp = Z[key]
            samples = []
            for ii in range(self.z_dim):
                z = z_temp.clone()
                for val in interp:
                    z[:, ii] = val
                    sample = torch.sigmoid(decoder(z)).data
                    samples.append(sample)
                    images.append(sample)
            samples = torch.cat(samples, dim=0)

        output_dir = os.path.join(self.output_dir, str(self.global_iter))
        os.makedirs(output_dir, exist_ok=True)
        images = torch.cat(images)
        images = images.view(
            len(Z), self.z_dim, len(interp), self.nchan, 64, 64
        ).transpose(1, 2)
        for i, key in enumerate(Z.keys()):
            for j, val in enumerate(interp):
                save_image(
                    tensor=images[i][j].cpu(),
                    filename=os.path.join(output_dir, "{}_{}.jpg".format(key, j)),
                    nrow=self.z_dim,
                    pad_value=1,
                )

    def save_ckpt(self, filename):
        """save_checkpoint

        used to save checkpoints, because not doing so is asking for trouble

        inputs: filename cooresponding to data_out name from main.py
        outputs: checkpoints specified directory

        """

        model_states = {"net": self.net.state_dict()}
        optim_states = {"optim": self.optim.state_dict()}

        states = {
            "iter": self.global_iter,
            "model_states": model_states,
            "optim_states": optim_states,
        }

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode="wb+") as f:
            torch.save(states, f)

    def load_ckpt(self, filename):
        """load_checkpoint

        loads checkpoints cooresdponding to run name
        inputs: name of file, corresponding to data_out name from main.py
        outputs: loads checkpoints, resumes training

        """

        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)

            self.global_iter = checkpoint["iter"]
            self.net.load_state_dict(checkpoint["model_states"]["net"])
            self.optim.load_state_dict(checkpoint["optim_states"]["optim"])
            print("=> loaded ckpt '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no ckpt at '{}'".format(file_path))

