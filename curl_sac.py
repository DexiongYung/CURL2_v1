import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from encoder import make_encoder
from contrastive_models import BYOL, SimCLR, CURL2, CURL
from contrastive_models.SimCLR import SIMCLR_projection_MLP
import data_augs as rad

LOG_FREQ = 10000

CURL_STR = "CURL"
CURL2_STR = "CURL2"
BYOL_STR = "BYOL"
SIMCLR_STR = "SIMCLR"

K_MEANS_STR = "k_means"
CENTROID_STR = "centroid"

CONTRASTIVE_METHODS = [CURL_STR, CURL2_STR, BYOL_STR, SIMCLR_STR]
CLUSTER_METHODS = [CENTROID_STR, K_MEANS_STR]


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
        is_2_encoder=False,
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

        if is_2_encoder:
            self.encoder_2 = copy.deepcopy(self.encoder).requires_grad_(False)

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
        self.is_contrast = any(word in self.mode for word in CONTRASTIVE_METHODS)
        self.is_cluster = any(word in self.mode for word in CLUSTER_METHODS)
        self.is_other_env = "other" in self.mode
        self.aug_obs_only = "aug_obs_only" in self.mode
        self.aug_next_only = "aug_next_only" in self.mode
        self.is_double_encoder = "2encoder" in self.mode
        self.is_translate = "translate" in self.mode

        aug_to_func = {
            "crop": dict(func=rad.random_crop, params=dict(out=self.image_size)),
            "grayscale": dict(func=rad.random_grayscale, params=dict(p=0.3)),
            "cutout": dict(func=rad.random_cutout, params=dict(min_cut=10, max_cut=30)),
            "cutout_color": dict(
                func=rad.random_cutout_color, params=dict(min_cut=10, max_cut=30)
            ),
            "flip": dict(func=rad.random_flip, params=dict(p=0.2)),
            "rotate": dict(func=rad.random_rotation, params=dict(p=0.3)),
            "rand_conv": dict(func=rad.random_convolution, params=dict()),
            "translate": dict(
                func=rad.random_translate, params=dict(size=self.image_size)
            ),
            "random_resize_crop": dict(func=rad.random_resize_crop, params=dict()),
            "kornia_jitter": dict(
                func=rad.kornia_color_jitter,
                params=dict(bright=0.4, contrast=0.4, satur=0.4, hue=0.5),
            ),
            "instdisc": dict(func=rad.instdisc, params=dict()),
            "no_aug": dict(func=rad.no_aug, params=dict()),
        }

        self.augs_funcs = {}
        for aug_name in self.data_augs.split("-"):
            assert aug_name in aug_to_func, "invalid data aug string"
            self.augs_funcs[aug_name] = aug_to_func[aug_name]

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
            is_2_encoder=self.is_double_encoder,
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

        if self.is_double_encoder:
            self.double_encoder_optimizer = torch.optim.Adam(
                list(self.actor.encoder_2.parameters())
                + list(self.actor.encoder.parameters()),
                lr=encoder_lr,
            )

        if self.encoder_type == "pixel":
            self.create_contrast_alg_and_optimizer(
                encoder_feature_dim=encoder_feature_dim, encoder_lr=encoder_lr
            )

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()

    def create_contrast_alg_and_optimizer(self, encoder_feature_dim, encoder_lr):
        if CURL2_STR in self.mode:
            self.contrast_model = CURL2(
                z_dim=encoder_feature_dim,
                batch_size=self.latent_dim,
                critic=self.critic,
                critic_target=self.critic_target,
                output_type="continuous",
            )

            self.contrast_optimizer = self.contrast_model.create_optimizer(
                lr=encoder_lr
            )
            self.contrast_model.to(device=self.device)
        elif CURL_STR in self.mode:
            # create CURL encoder (the 128 batch size is probably unnecessary)
            self.contrast_model = CURL(
                encoder_feature_dim,
                self.latent_dim,
                self.critic,
                self.critic_target,
                output_type="continuous",
            ).to(self.device)

            # optimizer for critic encoder for reconstruction loss
            self.contrast_optimizer = self.contrast_model.create_optimizer(
                lr=encoder_lr
            )
        elif BYOL_STR in self.mode:
            self.contrast_model = BYOL(
                z_dim=encoder_feature_dim,
                critic=self.critic,
                critic_target=self.critic_target,
            ).to(device=self.device)

            self.contrast_optimizer = self.contrast_model.create_optimizer(
                lr=encoder_lr
            )
        elif SIMCLR_STR in self.mode:
            self.contrast_model = SimCLR(
                z_dim=encoder_feature_dim, critic=self.critic
            ).to(device=self.device)

            self.contrast_optimizer = self.contrast_model.create_optimizer(
                lr=encoder_lr
            )

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

        try:
            self.contrast_model.train(training)
        except Exception as e:
            pass

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def reshape_obs_for_actor(self, obs):
        if obs.shape[-1] > self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)
        elif obs.shape[-1] < self.image_size:
            obs = utils.center_crop_image(image=obs, output_size=obs.shape[-1])
            obs = utils.center_translate(image=obs, size=self.image_size)

        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)

        return obs

    def reshape_obses_for_actor(self, obs):
        if obs.shape[-1] > self.image_size:
            obs = utils.center_crop_images(obs, self.image_size)
        elif obs.shape[-1] < self.image_size:
            obs = utils.center_crop_images(image=obs, output_size=obs.shape[-1])
            obs = utils.center_translates(image=obs, size=self.image_size)

        obs = torch.FloatTensor(obs).to(self.device)

        return obs

    def select_action(self, obs):
        obs = self.reshape_obs_for_actor(obs=obs)
        with torch.no_grad():
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        obs = self.reshape_obs_for_actor(obs=obs)
        with torch.no_grad():
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

    def log_pos_and_neg_dist(self, z_anchor, z_pos, L, step, suffix=""):
        if step % self.log_interval == 0:
            with torch.no_grad():
                b, _ = z_anchor.shape
                cos_sim_matrix = F.cosine_similarity(
                    z_anchor[:, :, None], z_pos.t()[None, :, :]
                )
                pos_sim_mu = cos_sim_matrix.diagonal().mean()
                mask = torch.eye(b, b).bool().to(device=z_anchor.get_device())
                cos_sim_matrix_negs = cos_sim_matrix.masked_fill_(mask, 0)
                neg_sum = cos_sim_matrix_negs.sum()
                neg_sim_mu = neg_sum / (b * b - b)

                L.log(f"train/pos_sample_avg_cos_similarity{suffix}", pos_sim_mu, step)
                L.log(f"train/neg_sample_avg_cos_similarity{suffix}", neg_sim_mu, step)

    def update_cpc(self, obs_anchor, obs_pos, L, step, obs_other_env=None):

        # time flips
        """
        time_pos = contrastive_kwargs["time_pos"]
        time_anchor= contrastive_kwargs["time_anchor"]
        obs_anchor = torch.cat((obs_anchor, time_anchor), 0)
        obs_pos = torch.cat((obs_anchor, time_pos), 0)
        """
        z_a = self.contrast_model.encode(obs_anchor)
        z_pos = self.contrast_model.encode(obs_pos, ema=True)

        self.log_pos_and_neg_dist(z_anchor=z_a, z_pos=z_pos, L=L, step=step)

        if obs_other_env is not None:
            z_other = self.contrast_model.encode(obs_other_env, ema=True)
        else:
            z_other = None

        logits = self.contrast_model.compute_logits(
            z_a, z_pos, use_other_env=self.is_other_env, z_other_env=z_other
        )
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)

        self.contrast_optimizer.zero_grad()
        loss.backward()
        self.contrast_optimizer.step()

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
        z_a = self.contrast_model.encode(obs_anchor)
        z_pos = self.contrast_model.encode(obs_pos)

        self.log_pos_and_neg_dist(z_anchor=z_a, z_pos=z_pos, L=L, step=step)

        logits = self.contrast_model.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)

        self.contrast_optimizer.zero_grad()
        loss.backward()
        self.contrast_optimizer.step()

        if step % self.log_interval == 0:
            L.log("train/NXTent_loss", loss, step)

    def update_BYOL(self, obs_anchor, obs_pos, L, step):
        z_a = self.contrast_model.encode(obs_anchor)
        z_pos = self.contrast_model.encode(obs_pos, ema=True)

        self.log_pos_and_neg_dist(z_anchor=z_a, z_pos=z_pos, L=L, step=step)

        loss = self.contrast_model.compute_L2_MSE(z_a=z_a, z_pos=z_pos).mean()

        self.contrast_optimizer.zero_grad()
        loss.backward()
        self.contrast_optimizer.step()

        if step % self.log_interval == 0:
            L.log("train/BYOL_L2_MSE_loss", loss, step)

    def update_centroid(self, obs_anchor, aug_obs_list, L, step):
        centroid = self.contrast_model.encode(x=obs_anchor)
        loss = 0

        for aug_obs in aug_obs_list:
            encoded_aug = self.contrast_model.encode(aug_obs, ema=True)
            loss += F.mse_loss(centroid, encoded_aug)

        self.contrast_optimizer.zero_grad()
        loss.backward()
        self.contrast_optimizer.step()

        if step % self.log_interval == 0:
            L.log("train/centroid_MSE_loss", loss, step)

    def update_double_encoder(self, obs_centroid, obs_cluster, L, step):
        self.actor.encoder_2.requires_grad_(True)
        centroid = self.critic_target.encoder(obs_centroid)
        point = self.actor.encoder(obs_cluster[0])

        loss = F.mse_loss(point, centroid)

        for aug_obs in obs_cluster[1:]:
            point = self.actor.encoder_2(aug_obs)
            loss += F.mse_loss(point, centroid)

        if step % self.log_interval == 0:
            L.log("train/centroid_MSE_loss", loss, step)

        self.double_encoder_optimizer.zero_grad()
        loss.backward()
        self.double_encoder_optimizer.step()

        self.actor.encoder_2.requires_grad_(False)

    def update(self, replay_buffer, L, step):
        if self.encoder_type == "pixel":
            if self.is_contrast:
                (
                    obs,
                    action,
                    reward,
                    next_obs,
                    not_done,
                    contrastive_kwargs,
                ) = replay_buffer.sample_contrastive(
                    use_translate=self.is_translate, use_other_env=self.is_other_env
                )
            elif self.is_cluster or self.is_double_encoder:
                (
                    obs,
                    action,
                    reward,
                    next_obs,
                    not_done,
                    aug_obs_list,
                ) = replay_buffer.sample_cluster(
                    aug_funcs=self.augs_funcs, use_translate=self.is_translate
                )
            else:
                obs, action, reward, next_obs, not_done = replay_buffer.sample_rad(
                    aug_funcs=self.augs_funcs,
                    aug_obs_only=self.aug_obs_only,
                    aug_next_only=self.aug_next_only,
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
                    net=self.contrast_model.online_projection,
                    target_net=self.contrast_model.target_projection,
                    tau=self.encoder_tau,
                )
            elif CURL2_STR in self.mode:
                utils.soft_update_params(
                    net=self.contrast_model.online_encoder,
                    target_net=self.contrast_model.target_encoder,
                    tau=self.encoder_tau,
                )

        if step % self.cpc_update_freq == 0:
            if self.is_cluster:
                if CENTROID_STR in self.mode:
                    self.update_centroid(
                        obs_anchor=obs, aug_obs_list=aug_obs_list, L=L, step=step
                    )
                elif K_MEANS_STR in self.mode:
                    raise NotImplementedError("K-means is not implemented yet")
            elif self.is_double_encoder:
                self.update_double_encoder(
                    obs_centroid=obs, obs_cluster=aug_obs_list, L=L, step=step
                )
            elif self.is_contrast:
                if CURL_STR in self.mode:
                    self.update_cpc(
                        obs_anchor=contrastive_kwargs["obs_anchor"],
                        obs_pos=contrastive_kwargs["obs_pos"],
                        L=L,
                        step=step,
                        obs_other_env=contrastive_kwargs["obs_other_env"],
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
        torch.save(
            self.contrast_model.state_dict(), "%s/curl_%s.pt" % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(torch.load("%s/actor_%s.pt" % (model_dir, step)))
        self.critic.load_state_dict(torch.load("%s/critic_%s.pt" % (model_dir, step)))
