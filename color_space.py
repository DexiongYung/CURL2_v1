import torch
import numpy as np
from skimage import color


def reshape_to_RGB(obs):
    image_size = obs.shape[-1]
    return obs.reshape(-1, 3, image_size, image_size)


def reshape_to_frame_stack(obs, frame_stack_sz: int):
    image_size = obs.shape[-1]
    return obs.reshape(-1, frame_stack_sz, image_size, image_size)


def RGB_to_YDbDr(obs_RGB):
    obs_RGB *= 255.0
    img = np.asarray(obs_RGB, np.uint8)
    img = img.transpose(0, 2, 3, 1)
    img = color.rgb2luv(img)
    img = img.transpose(0, 3, 1, 2)
    return img


def RGB_to_YUV(obs_RGB):
    obs_RGB *= 255.0
    img = np.asarray(obs_RGB, np.uint8)
    img = img.transpose(0, 2, 3, 1)
    img = color.rgb2yuv(img)
    img = img.transpose(0, 3, 1, 2)
    return img


def RGB_to_YIQ(obs_RGB):
    obs_RGB *= 255.0
    img = np.asarray(obs_RGB, np.uint8)
    img = img.transpose(0, 2, 3, 1)
    img = color.rgb2yiq(img)
    img = img.transpose(0, 3, 1, 2)
    return img


def RGB_to_LAB(obs_RGB):
    obs_RGB *= 255.0
    img = np.asarray(obs_RGB, np.uint8)
    img = img.transpose(0, 2, 3, 1)
    img = color.rgb2lab(img)
    img = img.transpose(0, 3, 1, 2)
    return img


def split_RGB_into_R_GB(obs):
    return torch.split(tensor=obs, split_size_or_sections=[1, 2], dim=1)


def R_GB_to_frame_stacked_R_GB(obs_R, obs_GB, num_imgs: int):
    image_size = obs_R.shape[-1]
    return obs_R.reshape(-1, num_imgs, image_size, image_size), obs_GB.reshape(
        -1, num_imgs * 2, image_size, image_size
    )
