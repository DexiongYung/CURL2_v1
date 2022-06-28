from re import I
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from encoder import make_encoder
import data_augs as rad

LOG_FREQ = 10000

CURL_STR = "CURL"
CURL2_STR = "CURL2"
BYOL_STR = "BYOL"
SIMCLR_STR = "SIMCLR"

CONTRASTIVE_METHODS = [CURL_STR, CURL2_STR, BYOL_STR, SIMCLR_STR]
V2_METHODS = [CURL2_STR, BYOL_STR, SIMCLR_STR]

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
    "random_resize_crop": dict(func=rad.random_resize_crop, params=dict()),
    "kornia_jitter": dict(
        func=rad.kornia_color_jitter,
        params=dict(bright=0.4, contrast=0.4, satur=0.4, hue=0.5),
    ),
    "instdisc": dict(func=rad.instdisc, params=dict()),
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


class SIMCLR_projection_MLP(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super(SIMCLR_projection_MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim),
            nn.BatchNorm1d(z_dim),
        )

    def forward(self, x):
        return self.net(x)


class SIMCLR(nn.Module):
    def __init__(self, z_dim: int, critic: Critic) -> None:
        super(SIMCLR, self).__init__()
        self.encoder = critic.encoder
        self.projection_head = SIMCLR_projection_MLP(z_dim=z_dim)

    def encode(self, x):
        return self.projection_head.forward(self.encoder(x))

    def compute_logits(self, anchor, pos):
        logits = nn.functional.cosine_similarity(
            anchor[:, :, None], pos.t()[None, :, :]
        )
        return logits


class BYOL_projection_MLP(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super(BYOL_projection_MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim),
        )

    def forward(self, x):
        return self.net(x)


class BYOL(nn.Module):
    def __init__(
        self,
        z_dim,
        critic,
        critic_target,
    ):
        super(BYOL, self).__init__()
        self.encoder = critic.encoder
        self.encoder_target = critic_target.encoder
        self.online_projection = BYOL_projection_MLP(z_dim=z_dim)
        self.target_projection = BYOL_projection_MLP(z_dim=z_dim)
        self.online_predict = BYOL_projection_MLP(z_dim=z_dim)

    def encode(self, x, detach=False, target=False):
        if target:
            with torch.no_grad():
                z_out = self.encoder_target(x)
                z_fin = self.target_projection(z_out)
        else:
            z_out = self.encoder(x)
            z_proj = self.online_projection(z_out)
            z_fin = self.online_predict(z_proj)

        if detach:
            z_fin = z_fin.detach()
        return z_fin

    def compute_L2_MSE(self, z_a, z_pos):
        z_a_norm = F.normalize(z_a, dim=-1, p=2)
        z_pos_norm = F.normalize(z_pos, dim=-1, p=2)
        return 2 - 2 * (z_a_norm * z_pos_norm).sum(dim=-1)


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
        mode="",
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
        self.mode = mode

        self.augs_funcs = {}
        for aug_name in self.data_augs.split("-"):
            assert aug_name in AUG_TO_FUNC, "invalid data aug string"
            self.augs_funcs[aug_name] = AUG_TO_FUNC[aug_name]

        print(f"Aug set: {self.augs_funcs}")
        print(f"Mode is: {self.mode}")

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
            if CURL_STR in self.mode:
                # create CURL encoder (the 128 batch size is probably unnecessary)
                self.CURL = CURL(
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

                self.cpc_optimizer = torch.optim.Adam(
                    self.CURL.parameters(), lr=encoder_lr
                )
            elif BYOL_STR in self.mode:
                self.BYOL = BYOL(
                    z_dim=encoder_feature_dim,
                    critic=self.critic,
                    critic_target=self.critic_target,
                ).to(device=device)

                self.encoder_optimizer = torch.optim.Adam(
                    list(self.critic.encoder.parameters())
                    + list(self.BYOL.online_projection.parameters())
                    + list(self.BYOL.online_predict.parameters()),
                    lr=encoder_lr,
                )
            elif SIMCLR_STR in self.mode:
                self.SIMCLR = SIMCLR(z_dim=encoder_feature_dim, critic=self.critic).to(
                    device=device
                )
                self.encoder_optimizer = torch.optim.Adam(
                    self.critic.encoder.parameters(), lr=encoder_lr
                )
                self.projection_optimizer = torch.optim.Adam(
                    self.SIMCLR.parameters(), lr=encoder_lr
                )
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        try:
            self.CURL.train(training)
        except Exception as e:
            pass

        try:
            self.BYOL.train(training)
        except Exception as e:
            pass

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def reshape_action(self, obs):
        if obs.shape[-1] > self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)
        elif obs.shape[-1] < self.image_size:
            obs = utils.center_crop_image(image=obs, output_size=obs.shape[-1])
            obs = utils.center_translate(image=obs, size=self.image_size)

        return obs

    def select_action(self, obs):
        obs = self.reshape_action(obs=obs)
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        obs = self.reshape_action(obs=obs)
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

    def update_cpc(self, obs_anchor, obs_pos, L, step, use_SE=False):

        # time flips
        """
        time_pos = contrastive_kwargs["time_pos"]
        time_anchor= contrastive_kwargs["time_anchor"]
        obs_anchor = torch.cat((obs_anchor, time_anchor), 0)
        obs_pos = torch.cat((obs_anchor, time_pos), 0)
        """
        z_a = self.CURL.encode(obs_anchor)
        z_pos = self.CURL.encode(obs_pos, ema=not use_SE)

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

    def update_SIMCLR(self, obs_anchor, obs_pos, L, step):

        # time flips
        """
        time_pos = contrastive_kwargs["time_pos"]
        time_anchor= contrastive_kwargs["time_anchor"]
        obs_anchor = torch.cat((obs_anchor, time_anchor), 0)
        obs_pos = torch.cat((obs_anchor, time_pos), 0)
        """
        z_a = self.SIMCLR.encode(obs_anchor)
        z_pos = self.SIMCLR.encode(obs_pos)

        logits = self.SIMCLR.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)

        self.encoder_optimizer.zero_grad()
        self.projection_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        self.projection_optimizer.step()

        if step % self.log_interval == 0:
            L.log("train/NCE_loss", loss, step)

    def update_BYOL(self, obs_anchor, obs_pos, L, step):
        z_a = self.BYOL.encode(obs_anchor)
        z_pos = self.BYOL.encode(obs_pos, target=True)

        loss = self.BYOL.compute_L2_MSE(z_a=z_a, z_pos=z_pos).mean()

        self.encoder_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()

        if step % self.log_interval == 0:
            L.log("train/BYOL_L2_MSE_loss", loss, step)

    def update(self, replay_buffer, L, step):
        if self.encoder_type == "pixel":
            if any(word in self.mode for word in CONTRASTIVE_METHODS):
                is_v2 = any(word in self.mode for word in V2_METHODS)
                is_unique = "unique" in self.mode
                (
                    obs,
                    action,
                    reward,
                    next_obs,
                    not_done,
                    contrastive_kwargs,
                ) = replay_buffer.sample_contrastive(use_v2=is_v2, use_unique=is_unique)
            else:
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

            if BYOL_STR in self.mode:
                utils.soft_update_params(
                    net=self.BYOL.online_projection,
                    target_net=self.BYOL.target_projection,
                    tau=self.encoder_tau,
                )

        if step % self.cpc_update_freq == 0:
            if CURL_STR in self.mode:
                self.update_cpc(
                    obs_anchor=contrastive_kwargs["obs_anchor"],
                    obs_pos=contrastive_kwargs["obs_pos"],
                    L=L,
                    step=step,
                    use_SE="SE" in self.mode,
                )
            elif BYOL_STR in self.mode:
                self.update_BYOL(
                    obs_anchor=contrastive_kwargs["obs_anchor"],
                    obs_pos=contrastive_kwargs["obs_pos"],
                    L=L,
                    step=step,
                )
            elif SIMCLR_STR in self.mode:
                self.update_SIMCLR(
                    obs_anchor=contrastive_kwargs["obs_anchor"],
                    obs_pos=contrastive_kwargs["obs_pos"],
                    L=L,
                    step=step,
                )

    def save(self, model_dir, step):
        torch.save(self.actor.state_dict(), "%s/actor_%s.pt" % (model_dir, step))
        torch.save(self.critic.state_dict(), "%s/critic_%s.pt" % (model_dir, step))

    def save_curl(self, model_dir, step):
        torch.save(self.CURL.state_dict(), "%s/curl_%s.pt" % (model_dir, step))

    def load(self, model_dir, step):
        self.actor.load_state_dict(torch.load("%s/actor_%s.pt" % (model_dir, step)))
        self.critic.load_state_dict(torch.load("%s/critic_%s.pt" % (model_dir, step)))
