import os
import copy
import pandas as pd
import torch
import argparse
import numpy as np
import data_augs as rad
from logger import Logger
from contrastive_models.SimCLR import SimCLR
from evaluation.model_loading import load_actor
from evaluation.evaluate import evaluate
from utils import make_agent, set_json_to_args, create_env_and_replay_buffer, eval_mode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--actor_ckpt_path", type=str)
    parser.add_argument("--total_steps", type=int, default=10000)
    parser.add_argument("--sample_probs", type=float, default=0.5)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="logs_post_contrastive")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    eval_interval = args.eval_interval

    os.makedirs(args.out_dir, exist_ok=True)
    args = set_json_to_args(args=args, config_path=args.config_file)

    env_name = args.domain_name + "_" + args.task_name
    exp_name = os.path.join(env_name, args.id, f"seed_{args.seed}")
    out_dir = os.path.join(args.out_dir, exp_name)

    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(
        f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    )
    env, replay_buffer = create_env_and_replay_buffer(args=args, device=device)
    agent = make_agent(
        obs_shape=(3 * args.frame_stack, args.image_size, args.image_size),
        action_shape=env.action_space.shape,
        args=args,
        device=device,
    )
    agent = load_actor(agent=agent, ckpt_path=args.actor_ckpt_path)

    L = Logger(out_dir, use_tb=False)

    simclr = SimCLR(z_dim=args.encoder_feature_dim, critic=agent.actor)
    simclr_optimizer = simclr.create_optimizer(lr=args.encoder_lr)

    # agent_EMA = copy.deepcopy(agent)
    # actor_EMA_optimizer = torch.optim.Adam(
    #     agent_EMA.actor.encoder.parameters(), lr=args.encoder_lr
    # )

    done = True
    augs_func = agent.augs_funcs

    for step in range(args.total_steps):
        if done:
            obs = env.reset()
            done = False

        with eval_mode(agent):
            sample = np.random.rand(1).item()
            if sample > args.sample_probs:
                action = agent.select_action(obs / 255.0)
            else:
                action = env.action_space.sample()

        if step > args.init_steps:
            (
                obses,
                actions,
                rewards,
                next_obses,
                not_dones,
                idxs,
            ) = replay_buffer.sample_rad(aug_funcs=augs_func, return_idxs=True)
            (
                aug_obses,
                actions,
                rewards,
                aug_next_obses,
                not_dones,
            ) = replay_buffer.sample_rad(
                aug_funcs=dict(
                    crop=dict(func=rad.random_crop, params=dict(out=args.image_size)),
                    instdisc=dict(func=rad.instdisc, params=dict()),
                ),
                idxs=idxs,
            )

            mu, _, _, log_std = agent.actor.forward(obs=obses)
            aug_mu, _, _, aug_log_std = agent.actor.forward(obs=aug_obses)

            dist_og = torch.normal(mean=mu, std=log_std.exp()).detach()
            dist_aug = torch.normal(mean=aug_mu, std=aug_log_std.exp())

            kl = torch.kl_div(input=dist_aug, target=dist_og)

            z_anchor = agent.actor.encoder(obses)
            z_pos = agent.actor.encoder(aug_obses)
            nce_loss = simclr.compute_NCE_loss(z_anchor=z_anchor, z_pos=z_pos)

            simclr_optimizer.zero_grad()
            (kl + nce_loss).backward()
            simclr_optimizer.step()

        if step % eval_interval == 0:
            evaluate(
                env=env,
                agent=agent,
                video=None,
                num_episodes=9,
                L=L,
                step=0,
                args=args,
                work_dir=args.out_dir,
            )

        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs


if __name__ == "__main__":
    main()
