import copy
import torch
import torch.nn as nn
from curl_sac import Actor, weight_init
import utils


class ColorConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, bias=False, padding=1)

    def forward(self, imgs):
        image_size = imgs.shape[-1]
        stack_sz = int(imgs.shape[1] / 3)
        imgs_rgb = imgs.reshape(-1, 3, image_size, image_size)
        imgs_conv = self.conv(imgs_rgb)

        return imgs_conv.reshape(-1, stack_sz * 3, image_size, image_size)


class SuperLeaner(nn.Module):
    def __init__(
        self, pretrained_actor: Actor, image_size: int, device: str, use_pretrain=True
    ) -> None:
        super().__init__()
        self.actor = pretrained_actor
        self.teacher_encoder = copy.deepcopy(self.actor.encoder)
        self.actor.trunk.requires_grad_(False)
        self.teacher_encoder.requires_grad_(False)

        if not use_pretrain:
            self.actor.encoder.apply(weight_init)

        self.image_size = image_size
        self.device = device

    def reshape_obs_for_actor(self, obs):
        if obs.shape[-1] > self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)
        elif obs.shape[-1] < self.image_size:
            obs = utils.center_crop_image(image=obs, output_size=obs.shape[-1])
            obs = utils.center_translate(image=obs, size=self.image_size)

        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)

        return obs

    def select_action(self, obs):
        obs_reshaped = self.reshape_obs_for_actor(obs=obs)
        mu, _, _, _ = self.actor.forward(obs=obs_reshaped)
        return mu.cpu().data.numpy().flatten()

    def encode(self, obs, teacher=False):
        if teacher:
            return self.teacher_encoder(obs)
        else:
            return self.actor.encoder(obs)

    def create_optimizer(self, lr):
        return torch.optim.Adam(
            list(self.actor.encoder.parameters()),
            lr=lr,
        )
