import os
from gym import make
import torch
import argparse
import data_augs as rad
from logger import Logger
from evaluation.evaluate import evaluate
from utils import set_json_to_args, create_env_and_replay_buffer, eval_mode, make_agent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default="test")
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--actor_ckpt_path", type=str)
    parser.add_argument("--loss", type=str, default="MSE")
    parser.add_argument("--gate", type=str, default="cross_entropy")
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="logs_super_learner")
    parser.add_argument("--out_ckpt", type=str, default="super_learner_checkpoints")
    args = parser.parse_args()
    return args


def main():
    # args
    args = parse_args()
    id = args.id

    os.makedirs(args.out_dir, exist_ok=True)
    args = set_json_to_args(args=args, config_path=args.config_file)

    # Environment setup with output dir
    env_name = args.domain_name + "_" + args.task_name
    exp_name = os.path.join(env_name, id, f"seed_{args.seed}")
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
    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=env.action_space.shape,
        args=args,
        device=device,
    )
    agent.actor.load_state_dict(torch.load(args.actor_ckpt_path))

    L = Logger(out_dir, use_tb=False)
    done = True

    for step in range(args.total_steps):
        if step > args.init_steps:
            pass

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
