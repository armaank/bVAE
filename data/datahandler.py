"""
datahandler.py

manages preprocessing and orginizaiton of all datasets

"""

import os

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


class bvaeImageFolder(ImageFolder):
    """bvaeImageFolder
    
    class to manage indexing into datasets

    """

    def __init__(self, root, transform=None):
        super(bvaeImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img


def getDataset(args):
    """getDataset
    
    manages datasets for trainer.py
    mostly path and directory management as well as some basic preprocessing
    inputs: cli args passed from main.py

    """

    name = args.dataset
    data_dir = args.data_dir
    batch_size = args.batch_size
    image_size = args.image_size
    num_workers = args.n_workers

    assert image_size == 64  # only supporting image size of 64

    if name.lower() == "3dchairs":
        root = os.path.join(data_dir, "3DChairs")

    else:
        root = os.path.join(data_dir, "CelebA")

    transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )
    train_dict = {"root": root, "transform": transform}
    data_set = bvaeImageFolder

    train_data = data_set(**train_dict)
    data_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return data_loader


if __name__ == "__main__":

    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    data = bvaeImageFolder("data/CelebA", transform)
    loader = DataLoader(
        data,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        drop_last=True,
    )

    images1 = iter(loader).next()

