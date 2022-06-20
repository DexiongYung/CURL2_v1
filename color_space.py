import torch


def reshape_to_RGB(obs):
    image_size = obs.shape[-1]
    return obs.reshape(-1, 3, image_size, image_size)


def reshape_to_frame_stack(obs, frame_stack_sz: int):
    image_size = obs.shape[-1]
    return obs.reshape(-1, frame_stack_sz, image_size, image_size)


def RGB_to_YDbDr(obs_RGB):
    r = obs_RGB[:, 0]
    g = obs_RGB[:, 1]
    b = obs_RGB[:, 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    db = -0.450 * r + -0.883 * g + 1.333 * b
    dr = -1.333 * r + 1.116 * g + 0.217 * b

    return torch.stack([y, db, dr], -3)


def RGB_to_DrYDb(obs_RGB):
    r = obs_RGB[:, 0]
    g = obs_RGB[:, 1]
    b = obs_RGB[:, 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    db = -0.450 * r + -0.883 * g + 1.333 * b
    dr = -1.333 * r + 1.116 * g + 0.217 * b

    return torch.stack([dr, y, db], -3)


def RGB_to_DbDrY(obs_RGB):
    r = obs_RGB[:, 0]
    g = obs_RGB[:, 1]
    b = obs_RGB[:, 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    db = -0.450 * r + -0.883 * g + 1.333 * b
    dr = -1.333 * r + 1.116 * g + 0.217 * b

    return torch.stack([db, dr, y], -3)


def split_RGB_into_R_GB(obs):
    return torch.split(tensor=obs, split_size_or_sections=[1, 2], dim=1)


def R_GB_to_frame_stacked_R_GB(obs_R, obs_GB, num_imgs: int):
    image_size = obs_R.shape[-1]
    return obs_R.reshape(-1, num_imgs, image_size, image_size), obs_GB.reshape(
        -1, num_imgs * 2, image_size, image_size
    )
