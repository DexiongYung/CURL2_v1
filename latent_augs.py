from termios import CRPRNT
import numpy as np
import torch


def identity(latents):
    return latents


def random_cutout(latents, min_cut=5, max_cut=15):
    """
        args:
        latents: torch.tensor shape (B,C,H,W)
        min / max cut: int, min / max size of cutout 
        returns np.array
    """
    device = latents.get_device()
    n, c, h, w = latents.shape
    w1 = np.random.randint(min_cut, max_cut, n)
    h1 = np.random.randint(min_cut, max_cut, n)

    cutouts = torch.zeros(n, c, h, w)
    for i, (latent, w11, h11) in enumerate(zip(latents, w1, h1)):
        cut_img = torch.clone(latent)
        cut_img[:, h11:h11 + h11, w11:w11 + w11] = 0
        cutouts[i] = cut_img
    return cutouts.to(device=device)


def center_translate_images(image, size):
    b, c, h, w = image.shape
    assert size >= h and size >= w
    outs = torch.zeros(b, c, size, size)
    h1 = (size - h) // 2
    w1 = (size - w) // 2
    outs[:, :, h1:h1 + h, w1:w1 + w] = torch.clone(image)
    return outs


def random_crop(imgs, out=64):
    """
        args:
        imgs: np.array shape (B,C,H,W)
        out: output size (e.g. 84)
        returns np.array
    """
    n, c, h, w = imgs.shape
    crop_max = h - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = torch.zeros(n, c, out, out)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):        
        cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
    return cropped


def center_random_crop(latents, out:int=60):
    b, c, h, w = latents.shape
    pad = (h - out) // 2
    device = latents.get_device()
    cropped_out = random_crop(imgs=latents, out=out)
    return torch.nn.functional.pad(cropped_out, (pad, pad, pad ,pad)).to(device)


def gaussian(latents, std:float=0.05):
    return latents + torch.normal(mean=0, std=std, size=latents.shape).to(latents.get_device())

 
def random_translate(imgs, size=64, return_random_idxs=False, h1s=None, w1s=None):
    n, c, h, w = imgs.shape
    assert size >= h and size >= w
    outs = torch.zeros((n, c, size, size), dtype=imgs.dtype)
    h1s = np.random.randint(0, size - h + 1, n) if h1s is None else h1s
    w1s = np.random.randint(0, size - w + 1, n) if w1s is None else w1s
    for out, img, h1, w1 in zip(outs, imgs, h1s, w1s):
        out[:, h1:h1 + h, w1:w1 + w] = img
    if return_random_idxs:  # So can do the same to another set of imgs.
        return outs, dict(h1s=h1s, w1s=w1s)
    return outs


latent_aug_to_func = {
    'center_crop': center_random_crop,
    'cutout' : random_cutout,
    'gaussian' : gaussian,
    'translate' : random_translate,
    'crop': random_crop,
    'no_augs' : identity
}