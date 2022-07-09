import os
import math
import copy
import torch
import torch.nn as nn
import argparse
import data_augs as rad
from logger import Logger
from evaluation.evaluate import evaluate
from utils import (
    set_json_to_args,
    create_env_and_replay_buffer,
    eval_mode,
    center_crop_image,
    center_translate,
)
from curl_sac import Actor, gaussian_logprob, squash


def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2.0 * x) - math.log(2.0)

    return torch.mean(_log_cosh(y_pred - y_true))


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)


LOSSES = dict(MSE=torch.nn.MSELoss(), L1=torch.nn.L1Loss(), cosh=LogCoshLoss())


class SuperLearnerActor(nn.Module):
    def __init__(
        self,
        encoder_feature_dim,
        image_size,
        pretrained_actor: Actor,
        device,
    ):
        super().__init__()
        self.super_learner = nn.Linear(encoder_feature_dim * 2, encoder_feature_dim)
        self.actor = pretrained_actor
        self.encoder2 = copy.deepcopy(self.actor.encoder)
        self.image_size = image_size
        self.device = device

    def forward_pretrain_actor_encoder(self, obs):
        return self.actor.encoder(obs)

    def forward_super_learner(self, obs, detach_encoder=False):
        x1 = self.actor.encoder(obs).detach()
        x2 = self.encoder2(obs)

        if detach_encoder:
            x2.detach()

        return self.super_learner(torch.cat((x1, x2), dim=1))

    def forward(self, obs, compute_pi=True, compute_log_pi=True):
        obs_enc = self.actor.encoder(obs)
        obs_enc2 = self.encoder2(obs)
        obs_SL = self.super_learner(torch.cat((obs_enc, obs_enc2), dim=1))

        mu, log_std = self.actor.trunk(obs_SL).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.actor.log_std_min + 0.5 * (
            self.actor.log_std_max - self.actor.log_std_min
        ) * (log_std + 1)

        self.actor.outputs["mu"] = mu
        self.actor.outputs["std"] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def reshape_obs_for_actor(self, obs):
        if obs.shape[-1] > self.image_size:
            obs = center_crop_image(obs, self.image_size)
        elif obs.shape[-1] < self.image_size:
            obs = center_crop_image(image=obs, output_size=obs.shape[-1])
            obs = center_translate(image=obs, size=self.image_size)

        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)

        return obs

    def select_action(self, obs):
        obs = self.reshape_obs_for_actor(obs=obs)
        with torch.no_grad():
            mu, _, _, _ = self.forward(obs, compute_pi=False, compute_log_pi=False)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        obs = self.reshape_obs_for_actor(obs=obs)
        with torch.no_grad():
            mu, pi, _, _ = self.forward(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def create_optimizer(self, lr):
        return torch.optim.Adam(
            list(self.encoder2.parameters()) + list(self.super_learner.parameters()),
            lr=lr,
        )

    def save(self, model_dir, step):
        torch.save(self.state_dict(), "%s/super_learner_%s.pt" % (model_dir, step))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--actor_ckpt_path", type=str)
    parser.add_argument("--loss", type=str, default="MSE")
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="logs_super_learner")
    parser.add_argument("--out_ckpt", type=str, default="super_learner_checkpoints")
    args = parser.parse_args()
    return args


def main():
    # args
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    args = set_json_to_args(args=args, config_path=args.config_file)

    # Environment setup with output dir
    env_name = args.domain_name + "_" + args.task_name
    exp_name = os.path.join(env_name, args.id, f"seed_{args.seed}")
    out_dir = os.path.join(args.out_dir, exp_name)
    checkpoint_dir = os.path.join(args.out_ckpt, exp_name)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set device
    device = torch.device(
        f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    )
    env, replay_buffer = create_env_and_replay_buffer(args=args, device=device)
    obs_shape = (3 * args.frame_stack, args.image_size, args.image_size)
    pretrained_actor = Actor(
        obs_shape=obs_shape,
        action_shape=env.action_space.shape,
        hidden_dim=args.hidden_dim,
        encoder_type="pixel",
        encoder_feature_dim=args.encoder_feature_dim,
        log_std_max=args.actor_log_std_max,
        log_std_min=args.actor_log_std_min,
        num_filters=args.num_filters,
        num_layers=args.num_layers,
    )
    pretrained_actor.load_state_dict(torch.load(args.actor_ckpt_path))
    agent = SuperLearnerActor(
        encoder_feature_dim=args.encoder_feature_dim,
        pretrained_actor=pretrained_actor,
        image_size=args.image_size,
        device=device,
    ).to(device)
    optimizer = agent.create_optimizer(lr=args.encoder_lr)
    loss_fn = LOSSES[args.loss]

    L = Logger(out_dir, use_tb=False)

    done = True
    augs_func = dict(rand_conv=dict(func=rad.random_convolution, params=dict()))

    for step in range(args.total_steps):
        if step > args.init_steps:
            (
                obses,
                _,
                _,
                _,
                _,
                aug_obs_list,
            ) = replay_buffer.sample_cluster(augs_func, use_translate=True)

            super_learner_loss = 0

            super_learner_loss += loss_fn(
                agent.forward_pretrain_actor_encoder(obses).detach(),
                agent.forward_super_learner(obses, detach_encoder=True),
            )

            for aug_obs in aug_obs_list:
                super_learner_loss += loss_fn(
                    agent.forward_pretrain_actor_encoder(obses).detach(),
                    agent.forward_super_learner(aug_obs),
                )

            optimizer.zero_grad()
            super_learner_loss.backward()
            optimizer.step()

            if step % args.log_interval == 0:
                L.log("train/MSE", super_learner_loss, step)

        if step % args.eval_interval == 0 and step > args.init_steps or step == 0:
            evaluate(
                env=env,
                agent=agent,
                video=None,
                num_episodes=9,
                L=L,
                step=step,
                args=args,
                work_dir=args.out_dir,
            )
            agent.save(model_dir=checkpoint_dir, step=None)

        if done:
            obs = env.reset()
            done = False

        with eval_mode(agent):
            if step > args.init_steps:
                action = agent.select_action(obs / 255.0)
            else:
                action = env.action_space.sample()

        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs


if __name__ == "__main__":
    main()
