import os
import time
import torch
import argparse
import data_augs as rad
from logger import Logger
import torch.nn.functional as F
from contrastive_models.SimCLR import SIMCLR_projection_MLP
from contrastive_models.MoCo import MoCo2_projection_MLP
from contrastive_models.BYOL import BYOL_projection_MLP
from evaluation.evaluate import evaluate
from super_learner.LogCoshLoss import LogCoshLoss
from super_learner.super_learner import SuperLeaner
from utils import set_json_to_args, create_env_and_replay_buffer, eval_mode, make_agent

PROJECTION_HEADS = dict(
    simCLR=SIMCLR_projection_MLP,
    MoCo2=MoCo2_projection_MLP,
    BYOL=BYOL_projection_MLP,
    identity=torch.nn.Identity,
)

LOSS_FNS = dict(MSE=F.mse_loss, L1=F.l1_loss, LogCosh=LogCoshLoss())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, default="./configs/cheetah_2enc_trans_RC.json"
    )
    parser.add_argument(
        "--actor_ckpt_path",
        type=str,
        default="./ckpt/cheetah_run/2_encoder_translate_RC/actor_99999.pt",
    )
    parser.add_argument("--num_iter", type=int, default=10)
    args = parser.parse_args()
    return args


def main():
    # args
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    args = set_json_to_args(args=args, config_path=args.config_file)

    env_name = args.domain_name + "_" + args.task_name
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)
    exp_name = os.path.join(
        env_name, str(args.id), args.sl_id, args.loss, f"seed_{args.seed}", ts
    )
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
    agent.actor.encoder = agent.actor.encoder_2

    augmentation = rad.no_aug
    L = Logger(out_dir, use_tb=False)

    for i in range(args.num_iter):
        evaluate(
            env=env,
            agent=agent,
            video=None,
            num_episodes=9,
            L=L,
            step=i,
            args=args,
            work_dir=out_dir,
            augs=augmentation,
        )


if __name__ == "__main__":
    main()
