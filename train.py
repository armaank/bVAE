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
        # self.objective = args.objective
        self.nchan = 3
        # self.dist = "gaussian"
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
        for y in self.data_loader:
            self.global_iter += 1
            bar.update(1)
            x = Variable(x, self.use_cuda)
            x_recon, mu, logvar = self.network(x)
            recon_loss = r_loss(x, x_recon, "gaussian")
            kl_div, dim_kl_div, mean_kl_div = kl_div(mu, logvar)

            loss = recon_loss + self.beta * kl_div

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if self.global_iter % 100 == 0:
                bar.write(
                    "[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}".format(
                        self.global_iter,
                        recon_loss.data.item(),
                        kl_div.data.item(),
                        mean_kl_div.data.item(),
                    )
                )

                var = logvar.exp().mean(0).data
                var_str = ""
                for j, var_j in enumerate(var):
                    var_str += "var{}:{:.4f} ".format(j + 1, var_j)
                bar.write(var_str)

        if self.global_iter % self.save_step == 0:
            self.save_checkpoint("last")
            bar.write("Saved checkpoint(iter:{})".format(self.global_iter))

        if self.global_iter % 50000 == 0:
            self.save_checkpoint(str(self.global_iter))

    


        def save_checkpoint(self, filename, silent=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        win_states = {'recon':self.win_recon,
                        'kld':self.win_kld,
                        'mu':self.win_mu,
                        'var':self.win_var,}
        states = {'iter':self.global_iter,
                    'win_states':win_states,
                    'model_states':model_states,
                    'optim_states':optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.win_recon = checkpoint['win_states']['recon']
            self.win_kld = checkpoint['win_states']['kld']
            self.win_var = checkpoint['win_states']['var']
            self.win_mu = checkpoint['win_states']['mu']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
