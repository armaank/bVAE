# β-VAE

Replicating results from β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. Specifically, we've attempted to reproduce fig. 1 and fig. 2 of [1] (3DChairs and CelebA)

You can find my full report ![here](docs/report/kohli_ece471_midterm.pdf).

## Requirements
```
python 3.xx
pytorch 1.3
torchvision
```

## Baisc Usage
To train an instance of a β-VAE with hyperparameters specified by [1]:
```
sh scripts/run_chairs.sh # for 3dchairs dataset
sh scripts/run_celeb.sh # for celebA dataset
```


## References
[1] I. Higgins, L. Matthey, A. Pal, C. Burgess, X. Glorot, M. M. Botvinick, S. Mohamed, and A. Lerchner,
“beta-vae: Learning basic visual concepts with a constrained variational framework,” in ICLR, 2017.

[2] D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” CoRR, vol. abs/1312.6114, 2013.

[3] C. P. Burgess, I. Higgins, A. Pal, L. Matthey, N. Watters, G. Desjardins, and A. Lerchner, “Understanding disentangling in beta-vae,” ArXiv, vol. abs/1804.03599, 2018.

## Credits
My thanks go to Ali for lending me time on his very fancy computer :)





