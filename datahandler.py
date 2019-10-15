"""
datahandler.py

downloads and processs datasets for betaVAE experiments

"""
import os
# original_umask = os.umask(0)
import subprocess
import tarfile
import glob

import numpy as np
from PIL import Image
from tqdm import tqdm  # might remove

# proj_dir = os.path.join(".")
# print(proj_dir)
data_dir = "./data"
# celebA_data = os.path.join(data_dir, "/celebA")
# chairs3D_data = os.path.join(data_dir, "/chairs3D")


def preprocess(root, size=(64, 64), img_format="JPEG", center_crop=None):
    """Preprocess a folder of images.
    Parameters
    ----------
    root : string
        Root directory of all images.
    size : tuple of int
        Size (width, height) to rescale the images. If `None` don't rescale.
    img_format : string
        Format to save the image in. Possible formats:
        https://pillow.readthedocs.io/en/3.1.x/handbook/image-file-formats.html.
    center_crop : tuple of int
        Size (width, height) to center-crop the images. If `None` don't center-crop.
    """
    imgs = []
    for ext in [".png", ".jpg", ".jpeg"]:
        imgs += glob.glob(os.path.join(root, "*" + ext))

    for img_path in tqdm(imgs):
        img = Image.open(img_path)
        width, height = img.size

        if size is not None and width != size[1] or height != size[0]:
            img = img.resize(size, Image.ANTIALIAS)

        if center_crop is not None:
            new_width, new_height = center_crop
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = (width + new_width) // 2
            bottom = (height + new_height) // 2

            img.crop((left, top, right, bottom))

        img.save(img_path, img_format)


def getChairs(data_dir):
    """ downloads, unpackages and preprocesses chairs3D dataset
    
    inputs: main data directory
    outputs: relevant chair data
   
    """

    urls = {
        "train": "https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar"
    }
    files = {"train": "chairs_64"}
    img_size = (1, 64, 64)
    # os.makedirs((os.path.join(data_dir, "/chairs"), 0777)
    save_path = os.path.join(data_dir, "/chairs/chairs.tar")
    subprocess.check_call(["curl", urls["train"], "--output", save_path])
    # tar = tarfile.open(save_path)
    # tar.extractall(os.path.join(data_dir, "/chairs"))
    # tar.close()
    # os.rename(
    #     os.path.join(os.path.join(data_dir, "/chairs"), "rendered_chairs"),
    #     "chairs_train",
    # )
    # preprocess(
    #     os.path.join(data_dir, "/chairs","/chairs_train", "*/*"), size=img_size[1:], center_crop=(400, 400)
    # )


if __name__ == "__main__":

    getChairs(data_dir)
