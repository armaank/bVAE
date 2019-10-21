"""
datahandler.py

handles all of the preprocessing and orginizaiton of all datasets

"""

import os

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


class bvaeImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(bvaeImageFolder, self).__init__(root, transform)

    def getImage(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img


class bvaeDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def getImage(self, index):
        return self.data_tensor[index]

    def getLength(self):
        return self.data_tensor.size(0)


def getDataset(args):
    name = args.datasets
    data_wd = args.dset_dir
    batch_size = args.batch_size
    image_size = args.image_size
    assert image_size == 64  # only supporting image size of 64

    if name.lower() == "3dchairs":
        root = os.path.join(data_wd, "3DChairs")

    elif name.lower == "celeba":
        root = os.path.join(data_wd, "CelebA")

    transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )
    train_dict = {"root": root, "transform": transform}
    data_set = bvaeImageFolder

    train_data = data_set(**train_dict)
    data_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True
    )

    return data_loader


if __name__ == "__main__":

    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    dset = bvaeImageFolder("data/CelebA", transform)
    loader = DataLoader(
        dset,
        batch_size=32,
        shuffle=True,
        num_workers=1,
        pin_memory=False,
        drop_last=True,
    )

    images1 = iter(loader).next()

