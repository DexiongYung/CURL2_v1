from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from encoder import make_encoder
import data_augs as rad
from color_space import (
    reshape_to_RGB,
    split_RGB_into_R_GB,
    R_GB_to_frame_stacked_R_GB,
    split_into_frame_stacked_R_GB,
)

LOG_FREQ = 10000

AUG_TO_FUNC = {
    "crop": dict(func=rad.random_crop, params=dict(out=84)),
    "grayscale": dict(func=rad.random_grayscale, params=dict(p=0.3)),
    "cutout": dict(func=rad.random_cutout, params=dict(min_cut=10, max_cut=30)),
    "cutout_color": dict(
        func=rad.random_cutout_color, params=dict(min_cut=10, max_cut=30)
    ),
    "flip": dict(func=rad.random_flip, params=dict(p=0.2)),
    "rotate": dict(func=rad.random_rotation, params=dict(p=0.3)),
    "rand_conv": dict(func=rad.random_convolution, params=dict()),
    "color_jitter": dict(
        func=rad.random_color_jitter,
        params=dict(bright=0.4, contrast=0.4, satur=0.4, hue=0.5),
    ),
    "translate": dict(func=rad.random_translate, params=dict()),
    "center_crop": dict(func=rad.center_random_crop, params=dict(out=84)),
    "translate_cc": dict(func=rad.translate_center_crop, params=dict(crop_sz=100)),
    "kornia_jitter": dict(
        func=rad.kornia_color_jitter,
        params=dict(bright=0.4, contrast=0.4, satur=0.4, hue=0.5),
    ),
    "in_frame_translate": dict(func=rad.in_frame_translate, params=dict(size=94)),
    "crop_translate": dict(func=rad.crop_translate, params=dict(size=100)),
    "center_crop_drac": dict(func=rad.center_crop_DrAC, params=dict(out=116)),
    "no_aug": dict(func=rad.no_aug, params=dict()),
}


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        hidden_dim,
        encoder_type,
        encoder_feature_dim,
        log_std_min,
        log_std_max,
        num_layers,
        num_filters,
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type,
            obs_shape,
            encoder_feature_dim,
            num_layers,
            num_filters,
            output_logits=True,
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0]),
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )

        self.outputs["mu"] = mu
        self.outputs["std"] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram("train_actor/%s_hist" % k, v, step)

        L.log_param("train_actor/fc1", self.trunk[0], step)
        L.log_param("train_actor/fc2", self.trunk[2], step)
        L.log_param("train_actor/fc3", self.trunk[4], step)


class QFunction(nn.Module):
    """MLP for q-function."""

    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        hidden_dim,
        encoder_type,
        encoder_feature_dim,
        num_layers,
        num_filters,
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type,
            obs_shape,
            encoder_feature_dim,
            num_layers,
            num_filters,
            output_logits=True,
        )

        self.Q1 = QFunction(self.encoder.feature_dim, action_shape[0], hidden_dim)
        self.Q2 = QFunction(self.encoder.feature_dim, action_shape[0], hidden_dim)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs["q1"] = q1
        self.outputs["q2"] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram("train_critic/%s_hist" % k, v, step)

        for i in range(3):
            L.log_param("train_critic/q1_fc%d" % i, self.Q1.trunk[i * 2], step)
            L.log_param("train_critic/q2_fc%d" % i, self.Q2.trunk[i * 2], step)


class CURL(nn.Module):
    """
    CURL
    """

    def __init__(
        self,
        obs_shape,
        z_dim,
        batch_size,
        critic,
        critic_target,
        output_type="continuous",
    ):
        super(CURL, self).__init__()
        self.batch_size = batch_size

        self.encoder = critic.encoder

        self.encoder_target = critic_target.encoder

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

    def encode(self, x, detach=False, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()
        return z_out

    # def update_target(self):
    #    utils.soft_update_params(self.encoder, self.encoder_target, 0.05)

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits


class RadSacAgent(object):
    """RAD with SAC."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.1,
        alpha_lr=1e-4,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type="pixel",
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.05,
        num_layers=4,
        num_filters=32,
        cpc_update_freq=1,
        log_interval=100,
        detach_encoder=False,
        latent_dim=128,
        data_augs="",
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.latent_dim = latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type
        self.data_augs = data_augs

        self.augs_funcs = {}
        for aug_name in self.data_augs.split("-"):
            assert aug_name in AUG_TO_FUNC, "invalid data aug string"
            self.augs_funcs[aug_name] = AUG_TO_FUNC[aug_name]

        print(f"Aug set: {self.augs_funcs}")

        self.actor = Actor(
            obs_shape,
            action_shape,
            hidden_dim,
            encoder_type,
            encoder_feature_dim,
            actor_log_std_min,
            actor_log_std_max,
            num_layers,
            num_filters,
        ).to(device)

        self.critic = Critic(
            obs_shape,
            action_shape,
            hidden_dim,
            encoder_type,
            encoder_feature_dim,
            num_layers,
            num_filters,
        ).to(device)

        self.critic_target = Critic(
            obs_shape,
            action_shape,
            hidden_dim,
            encoder_type,
            encoder_feature_dim,
            num_layers,
            num_filters,
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic, and CURL and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        if self.encoder_type == "pixel":
            # create CURL encoder (the 128 batch size is probably unnecessary)
            self.CURL = CURL(
                obs_shape,
                encoder_feature_dim,
                self.latent_dim,
                self.critic,
                self.critic_target,
                output_type="continuous",
            ).to(self.device)

            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

            self.cpc_optimizer = torch.optim.Adam(self.CURL.parameters(), lr=encoder_lr)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder_type == "pixel":
            self.CURL.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)

        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(
            obs, action, detach_encoder=self.detach_encoder
        )
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        if step % self.log_interval == 0:
            L.log("train_critic/loss", critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # TODD!!!
        # self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log("train_actor/loss", actor_loss, step)
            L.log("train_actor/target_entropy", self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(
            dim=-1
        )
        if step % self.log_interval == 0:
            L.log("train_actor/entropy", entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # TODO!!!
        # self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log("train_alpha/loss", alpha_loss, step)
            L.log("train_alpha/value", self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_cpc(self, obs_anchor, obs_pos, L, step):

        # time flips
        """
        time_pos = cpc_kwargs["time_pos"]
        time_anchor= cpc_kwargs["time_anchor"]
        obs_anchor = torch.cat((obs_anchor, time_anchor), 0)
        obs_pos = torch.cat((obs_anchor, time_pos), 0)
        """
        z_a = self.CURL.encode(obs_anchor)
        z_pos = self.CURL.encode(obs_pos, ema=True)

        logits = self.CURL.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)

        self.encoder_optimizer.zero_grad()
        self.cpc_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.cpc_optimizer.step()
        if step % self.log_interval == 0:
            L.log("train/curl_loss", loss, step)

    def update(self, replay_buffer, L, step):
        if self.encoder_type == "pixel":
            obs, action, reward, next_obs, not_done = replay_buffer.sample_rad(
                self.augs_funcs
            )
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio()

        if step % self.log_interval == 0:
            L.log("train/batch_reward", reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder, self.encoder_tau
            )

        # if step % self.cpc_update_freq == 0 and self.encoder_type == 'pixel':
        #    obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
        #    self.update_cpc(obs_anchor, obs_pos, L, step)

    def save(self, model_dir, step):
        torch.save(self.actor.state_dict(), "%s/actor_%s.pt" % (model_dir, step))
        torch.save(self.critic.state_dict(), "%s/critic_%s.pt" % (model_dir, step))

    def save_curl(self, model_dir, step):
        torch.save(self.CURL.state_dict(), "%s/curl_%s.pt" % (model_dir, step))

    def load(self, model_dir, step):
        self.actor.load_state_dict(torch.load("%s/actor_%s.pt" % (model_dir, step)))
        self.critic.load_state_dict(torch.load("%s/critic_%s.pt" % (model_dir, step)))


def make_actor_critic_critic_target(
    obs_shape,
    action_shape,
    hidden_dim,
    encoder_type,
    encoder_feature_dim,
    actor_log_std_min,
    actor_log_std_max,
    num_layers,
    num_filters,
    device,
):
    actor = Actor(
        obs_shape,
        action_shape,
        hidden_dim,
        encoder_type,
        encoder_feature_dim,
        actor_log_std_min,
        actor_log_std_max,
        num_layers,
        num_filters,
    ).to(device)

    critic = Critic(
        obs_shape,
        action_shape,
        hidden_dim,
        encoder_type,
        encoder_feature_dim,
        num_layers,
        num_filters,
    ).to(device)

    critic_target = Critic(
        obs_shape,
        action_shape,
        hidden_dim,
        encoder_type,
        encoder_feature_dim,
        num_layers,
        num_filters,
    ).to(device)

    return actor, critic, critic_target


class InfoMinSacAgent(object):
    """InfoMin with SAC."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.1,
        alpha_lr=1e-4,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type="pixel",
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.05,
        num_layers=4,
        num_filters=32,
        cpc_update_freq=1,
        log_interval=100,
        detach_encoder=False,
        latent_dim=128,
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.latent_dim = latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type

        num_frames = int(obs_shape[0] / 3)
        obs_shape_ch1 = (num_frames, self.image_size, self.image_size)
        obs_shape_ch23 = (num_frames * 2, self.image_size, self.image_size)

        (
            self.actor_ch1,
            self.critic_ch1,
            self.critic_target_ch1,
        ) = make_actor_critic_critic_target(
            obs_shape=obs_shape_ch1,
            action_shape=action_shape,
            hidden_dim=hidden_dim,
            encoder_type=encoder_type,
            encoder_feature_dim=encoder_feature_dim,
            actor_log_std_min=actor_log_std_min,
            actor_log_std_max=actor_log_std_max,
            num_layers=num_layers,
            num_filters=num_filters,
            device=device,
        )

        (
            self.actor_ch23,
            self.critic_ch23,
            self.critic_target_ch23,
        ) = make_actor_critic_critic_target(
            obs_shape=obs_shape_ch23,
            action_shape=action_shape,
            hidden_dim=hidden_dim,
            encoder_type=encoder_type,
            encoder_feature_dim=encoder_feature_dim,
            actor_log_std_min=actor_log_std_min,
            actor_log_std_max=actor_log_std_max,
            num_layers=num_layers,
            num_filters=num_filters,
            device=device,
        )

        # self.f1 = make_encoder(
        #     encoder_type=encoder_type,
        #     obs_shape=obs_shape_ch1,
        #     feature_dim=encoder_feature_dim,
        #     num_layers=num_layers,
        #     num_filters=num_filters,
        # ).to(self.device)
        # self.f2 = make_encoder(
        #     encoder_type=encoder_type,
        #     obs_shape=obs_shape_ch23,
        #     feature_dim=encoder_feature_dim,
        #     num_layers=num_layers,
        #     num_filters=num_filters,
        # ).to(self.device)
        # self.score_queue = deque(maxlen=2)
        # self.NCE_scale = 1

        # self.g = nn.Sequential(
        #     nn.Conv2d(obs_shape[0], obs_shape[0], 1).to(self.device), nn.ReLU()
        # )
        self.W_ch1 = nn.Parameter(
            torch.rand(encoder_feature_dim, encoder_feature_dim).to(self.device),
            requires_grad=True,
        )
        self.W_ch23 = nn.Parameter(
            torch.rand(encoder_feature_dim, encoder_feature_dim).to(self.device),
            requires_grad=True,
        )

        self.W_ch1_optimizer = torch.optim.Adam([self.W_ch1], lr=encoder_lr)
        self.W_ch23_optimizer = torch.optim.Adam([self.W_ch23], lr=encoder_lr)

        self.critic_encoders_optimizer = torch.optim.Adam(
            list(self.critic_ch1.encoder.parameters())
            + list(self.critic_ch23.encoder.parameters()),
            lr=encoder_lr,
        )
        self.critic_target_encoders_optimizer = torch.optim.Adam(
            list(self.critic_target_ch1.encoder.parameters())
            + list(self.critic_target_ch23.encoder.parameters()),
            lr=encoder_lr,
        )

        self.critic_target_ch1.load_state_dict(self.critic_ch1.state_dict())
        self.critic_target_ch23.load_state_dict(self.critic_ch23.state_dict())

        # tie encoders between actor and critic, and CURL and critic
        self.actor_ch1.encoder.copy_conv_weights_from(self.critic_ch1.encoder)
        self.actor_ch23.encoder.copy_conv_weights_from(self.critic_ch23.encoder)

        self.log_alpha_ch1 = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha_ch1.requires_grad = True
        self.log_alpha_ch23 = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha_ch23.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_ch1_optimizer = torch.optim.Adam(
            self.actor_ch1.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_ch1_optimizer = torch.optim.Adam(
            self.critic_ch1.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_ch1_optimizer = torch.optim.Adam(
            [self.log_alpha_ch1], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        # self.f1_optimizer = torch.optim.Adam(self.f1.parameters(), lr=encoder_lr / 2)

        self.actor_ch23_optimizer = torch.optim.Adam(
            self.actor_ch23.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_ch23_optimizer = torch.optim.Adam(
            self.critic_ch23.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_ch23_optimizer = torch.optim.Adam(
            [self.log_alpha_ch23], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        # self.f2_optimizer = torch.optim.Adam(self.f2.parameters(), lr=encoder_lr / 2)

        # self.g_optimizer = torch.optim.Adam(self.g.parameters(), lr=encoder_lr)
        # self.g_optimizer = torch.optim.lr_scheduler.CosineAnnealingLR(self.g_optimizer)

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()

    def train(self, training=True):
        self.critic_target_ch1.train()
        self.critic_target_ch23.train()
        self.training = training
        self.actor_ch1.train(training)
        self.critic_ch1.train(training)
        self.actor_ch23.train(training)
        self.critic_ch23.train(training)
        # self.f1.train(training)
        # self.f2.train(training)
        # self.g.train(training)

    @property
    def alpha_ch1(self):
        return self.log_alpha_ch1.exp()

    @property
    def alpha_ch23(self):
        return self.log_alpha_ch23.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            obs = rad.YDbDr(imgs=obs)
            # obs = self.g(obs)
            obs_R, obs_GB = split_into_frame_stacked_R_GB(obs=obs)
            mu_ch1, _, _, _ = self.actor_ch1(
                obs_R, compute_pi=False, compute_log_pi=False
            )
            mu_ch23, _, _, _ = self.actor_ch23(
                obs_GB, compute_pi=False, compute_log_pi=False
            )
            mu_avg = (mu_ch1 + mu_ch23) / 2
            return mu_avg.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)

        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            obs = rad.YDbDr(imgs=obs)
            # obs = self.g(obs)
            obs_R, obs_GB = split_into_frame_stacked_R_GB(obs=obs)
            _, pi_ch1, _, _ = self.actor_ch1(obs_R, compute_log_pi=False)
            _, pi_ch23, _, _ = self.actor_ch23(obs_GB, compute_log_pi=False)
            pi_avg = (pi_ch1 + pi_ch23) / 2
            return pi_avg.cpu().data.numpy().flatten()

    def calculate_critic_loss(
        self,
        obs,
        action,
        reward,
        next_obs,
        not_done,
        actor,
        critic,
        critic_target,
        alpha,
    ):
        with torch.no_grad():
            _, policy_action, log_pi, _ = actor(next_obs)
            target_Q1, target_Q2 = critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = critic(obs, action, detach_encoder=self.detach_encoder)
        return F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

    def update_critic(
        self,
        obs_ch1,
        obs_ch23,
        action,
        reward,
        next_obs_ch1,
        next_obs_ch23,
        not_done,
        L,
        step,
    ):
        ch1_critic_loss = self.calculate_critic_loss(
            obs=obs_ch1,
            action=action,
            reward=reward,
            next_obs=next_obs_ch1,
            not_done=not_done,
            actor=self.actor_ch1,
            critic=self.critic_ch1,
            critic_target=self.critic_target_ch1,
            alpha=self.alpha_ch1,
        )

        ch23_critic_loss = self.calculate_critic_loss(
            obs=obs_ch23,
            action=action,
            reward=reward,
            next_obs=next_obs_ch23,
            not_done=not_done,
            actor=self.actor_ch23,
            critic=self.critic_ch23,
            critic_target=self.critic_target_ch23,
            alpha=self.alpha_ch23,
        )

        # f1_out = self.f1(obs_ch1)
        # f2_out = self.f2(obs_ch23)

        # logits = torch.matmul(
        #     self.critic_ch1.encoder(obs_ch1), self.critic_ch23.encoder(obs_ch23).T
        # )
        # logits = logits - torch.max(logits, 1)[0][:, None]
        # labels = torch.arange(logits.shape[0]).long().to(self.device)
        # NCE_loss = self.cross_entropy_loss(logits, labels)

        # cpc_loss_1 = self.calculate_cpc_loss(
        #     obs_anchor=obs_ch1,
        #     obs_pos=obs_ch23,
        #     critic=self.critic_ch1,
        #     critic_target=self.critic_target_ch23,
        #     W=self.W_ch1,
        # )
        # cpc_loss_2 = self.calculate_cpc_loss(
        #     obs_anchor=obs_ch23,
        #     obs_pos=obs_ch1,
        #     critic=self.critic_ch23,
        #     critic_target=self.critic_target_ch1,
        #     W=self.W_ch23,
        # )

        if step % self.log_interval == 0:
            L.log("train_critic/ch1_loss", ch1_critic_loss, step)
            L.log("train_critic/ch23_loss", ch23_critic_loss, step)
            # L.log("train_critic/CPC_loss_1", cpc_loss_1, step)
            # L.log("train_critic/CPC_loss_2", cpc_loss_2, step)

        # if len(self.score_queue) > 1:
        #     score_diff = self.score_queue[1] - self.score_queue[0]
        #     self.NCE_scale = self.NCE_scale if score_diff >= 0 else -1 * self.NCE_scale

        # Optimize the critic
        self.critic_ch1_optimizer.zero_grad()
        self.critic_ch23_optimizer.zero_grad()
        # self.W_ch1_optimizer.zero_grad()
        # self.W_ch23_optimizer.zero_grad()
        # self.g_optimizer.zero_grad()
        (ch1_critic_loss + ch23_critic_loss).backward()
        self.critic_ch1_optimizer.step()
        self.critic_ch23_optimizer.step()
        # self.W_ch1_optimizer.step()
        # self.W_ch23_optimizer.step()
        # self.g_optimizer.step()

        # f1_out = self.f1(obs_ch1.detach())
        # f2_out = self.f2(obs_ch23.detach())

        # logits = torch.matmul(f1_out, f2_out.T)
        # logits = logits - torch.max(logits, 1)[0][:, None]
        # max_NCE_loss = -1 * self.cross_entropy_loss(logits, labels)

        # self.f1_optimizer.zero_grad()
        # self.f2_optimizer.zero_grad()
        # (max_NCE_loss).backward()
        # self.f1_optimizer.step()
        # self.f2_optimizer.step()

    def update_cpc(self, obs_ch1, obs_ch23, step, L):
        cpc_loss_1 = self.calculate_cpc_loss(
            obs_anchor=obs_ch1,
            obs_pos=obs_ch23,
            critic=self.critic_ch1,
            critic_target=self.critic_target_ch23,
            W=self.W_ch1,
        )
        cpc_loss_2 = self.calculate_cpc_loss(
            obs_anchor=obs_ch23,
            obs_pos=obs_ch1,
            critic=self.critic_ch23,
            critic_target=self.critic_target_ch1,
            W=self.W_ch23,
        )

        self.critic_encoders_optimizer.zero_grad()
        self.critic_target_encoders_optimizer.zero_grad()
        (cpc_loss_1 + cpc_loss_2).backward()
        self.critic_encoders_optimizer.step()
        self.critic_target_encoders_optimizer.step()

        if step % self.log_interval == 0:
            L.log("train_critic/CPC_loss_1", cpc_loss_1, step)
            L.log("train_critic/CPC_loss_2", cpc_loss_2, step)

    def calcuate_actor_loss(self, obs, actor, critic, alpha):
        _, pi, log_pi, log_std = actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (alpha.detach() * log_pi - actor_Q).mean()
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(
            dim=-1
        )

        return actor_loss, entropy, log_pi

    def calculate_NCE_loss(self, obs_ch1, obs_ch23, step, L):
        f1_out = self.f1(obs_ch1)
        f2_out = self.f2(obs_ch23)

        logits = torch.matmul(
            self.critic_ch1.encoder(obs_ch1), self.critic_ch23.encoder(obs_ch23).T
        )
        logits = logits - torch.max(logits, 1)[0][:, None]
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        NCE_loss = self.cross_entropy_loss(logits, labels)

        if step % self.log_interval == 0:
            L.log("train_critic/NCE_loss", NCE_loss, step)

        f1_out = self.f1(obs_ch1.detach())
        f2_out = self.f2(obs_ch23.detach())

        logits = torch.matmul(f1_out, f2_out.T)
        logits = logits - torch.max(logits, 1)[0][:, None]
        max_NCE_loss = -1 * self.cross_entropy_loss(logits, labels)

        return NCE_loss, max_NCE_loss

    def compute_logits(self, W, z_pos, z_a):
        Wz = torch.matmul(W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

    def calculate_cpc_loss(
        self, obs_anchor, obs_pos, critic: Critic, critic_target: Critic, W
    ):
        logits = self.compute_logits(
            W=W,
            z_pos=critic_target.encoder(obs_pos).detach(),
            z_a=critic.encoder(obs_anchor),
        )
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        return self.cross_entropy_loss(logits, labels)

    def update_InfoMin(self, obs_ch1, obs_ch23, step, L):
        f1_out = self.f1(obs_ch1)
        f2_out = self.f2(obs_ch23)

        logits = torch.matmul(
            self.critic_ch1.encoder(obs_ch1), self.critic_ch23.encoder(obs_ch23).T
        )
        logits = logits - torch.max(logits, 1)[0][:, None]
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        NCE_loss = self.cross_entropy_loss(logits, labels)

        if step % self.log_interval == 0:
            L.log("train_critic/NCE_loss", NCE_loss, step)

        # optimize the actor
        self.g_optimizer.zero_grad()
        NCE_loss.backward()
        self.g_optimizer.step()

        f1_out = self.f1(obs_ch1.detach())
        f2_out = self.f2(obs_ch23.detach())

        logits = torch.matmul(f1_out, f2_out.T)
        logits = logits - torch.max(logits, 1)[0][:, None]
        max_NCE_loss = -1 * self.cross_entropy_loss(logits, labels)

        self.f1_optimizer.zero_grad()
        self.f2_optimizer.zero_grad()
        (max_NCE_loss).backward()
        self.f1_optimizer.step()
        self.f2_optimizer.step()

    def update_actor_and_alpha(self, obs_ch1, obs_ch23, L, step):
        # detach encoder, so we don't update it with the actor loss
        ch1_actor_loss, ch1_entropy, ch1_log_pi = self.calcuate_actor_loss(
            obs=obs_ch1,
            actor=self.actor_ch1,
            critic=self.critic_ch1,
            alpha=self.alpha_ch1,
        )
        ch23_actor_loss, ch23_entropy, ch23_log_pi = self.calcuate_actor_loss(
            obs=obs_ch23,
            actor=self.actor_ch23,
            critic=self.critic_ch23,
            alpha=self.alpha_ch23,
        )

        if step % self.log_interval == 0:
            L.log("train_actor/ch1_loss", ch1_actor_loss, step)
            L.log("train_actor/ch1_entropy", ch1_entropy.mean(), step)
            L.log("train_actor/ch23_loss", ch23_actor_loss, step)
            L.log("train_actor/ch23_entropy", ch23_entropy.mean(), step)

        # optimize the actor
        self.actor_ch1_optimizer.zero_grad()
        self.actor_ch23_optimizer.zero_grad()
        (ch1_actor_loss + ch23_actor_loss).backward()
        self.actor_ch1_optimizer.step()
        self.actor_ch23_optimizer.step()

        # TODO!!!
        # self.actor.log(L, step)

        self.log_alpha_ch1_optimizer.zero_grad()
        self.log_alpha_ch23_optimizer.zero_grad()
        alpha_ch1_loss = (
            self.alpha_ch1 * (-ch1_log_pi - self.target_entropy).detach()
        ).mean()
        alpha_ch23_loss = (
            self.alpha_ch23 * (-ch23_log_pi - self.target_entropy).detach()
        ).mean()
        if step % self.log_interval == 0:
            L.log("train_alpha/ch1_loss", alpha_ch1_loss, step)
            L.log("train_alpha/ch1_value", self.alpha_ch1, step)
            L.log("train_alpha/ch23_loss", alpha_ch23_loss, step)
            L.log("train_alpha/ch23_value", self.alpha_ch23, step)
        (alpha_ch1_loss + alpha_ch23_loss).backward()
        self.log_alpha_ch1_optimizer.step()
        self.log_alpha_ch23_optimizer.step()

    def update(self, replay_buffer, L, step):
        if self.encoder_type == "pixel":
            obs, action, reward, next_obs, not_done = replay_buffer.sample_rad(None)
            obs = rad.YDbDr(obs)
            next_obs = rad.YDbDr(next_obs)
            # obs = self.g(obs)
            # next_obs = self.g(obs)
            obs_R, obs_GB = split_into_frame_stacked_R_GB(obs=obs)
            next_obs_R, next_obs_GB = split_into_frame_stacked_R_GB(obs=next_obs)
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio()

        if step % self.log_interval == 0:
            L.log("train/batch_reward", reward.mean(), step)

        self.update_critic(
            obs_ch1=obs_R.detach(),
            obs_ch23=obs_GB.detach(),
            action=action,
            reward=reward,
            next_obs_ch1=next_obs_R,
            next_obs_ch23=next_obs_GB,
            not_done=not_done,
            L=L,
            step=step,
        )

        # self.update_InfoMin(obs_ch1=obs_R, obs_ch23=obs_GB, step=step, L=L)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(
                obs_ch1=obs_R.detach(), obs_ch23=obs_GB.detach(), L=L, step=step
            )

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic_ch1.Q1, self.critic_target_ch1.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic_ch1.Q2, self.critic_target_ch1.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic_ch1.encoder,
                self.critic_target_ch1.encoder,
                self.encoder_tau,
            )
            utils.soft_update_params(
                self.critic_ch23.Q1, self.critic_target_ch23.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic_ch23.Q2, self.critic_target_ch23.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic_ch23.encoder,
                self.critic_target_ch23.encoder,
                self.encoder_tau,
            )

        self.update_cpc(obs_ch1=obs_R, obs_ch23=obs_GB, step=step, L=L)

    def save(self, model_dir, step):
        torch.save(self.actor.state_dict(), "%s/actor_%s.pt" % (model_dir, step))
        torch.save(self.critic.state_dict(), "%s/critic_%s.pt" % (model_dir, step))

    def load(self, model_dir, step):
        self.actor.load_state_dict(torch.load("%s/actor_%s.pt" % (model_dir, step)))
        self.critic.load_state_dict(torch.load("%s/critic_%s.pt" % (model_dir, step)))
