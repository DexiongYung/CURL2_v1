import os
import torch
import argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
from numpy import reshape
from logger import Logger
from evaluation.model_loading import load_actor
from evaluation.evaluate import evaluate
from utils import make_agent, set_json_to_args, create_env_and_replay_buffer, eval_mode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--actor_ckpt_path", type=str)
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--sample_percent", type=float, default=1)
    parser.add_argument("--out_dir", type=str, default="t_SNE")
    args = parser.parse_args()
    return args


def create_PCA_subplot(z_tensor, index: int, title: str, axes):
    # tsne = TSNE(n_components=2, random_state=123)
    # z_enc = tsne.fit_transform(z_tensor.detach().cpu().numpy())
    pca = PCA(n_components=2)
    z_enc = pca.fit_transform(z_tensor.detach().cpu().numpy())
    # sns.scatterplot(data=z_enc, ax=axes[index])
    axes[index].scatter(x=z_enc[:, 0], y=z_enc[:, 1])
    axes[index].set_title(title)


def create_t_SNE_subplot(z_tensor, index: int, title: str, axes):
    tsne = TSNE(n_components=2, random_state=123)
    z_enc = tsne.fit_transform(z_tensor.detach().cpu().numpy())
    axes[index].scatter(x=z_enc[:, 0], y=z_enc[:, 1])
    axes[index].set_title(title)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    args = set_json_to_args(args=args, config_path=args.config_file)

    env_name = args.domain_name + "_" + args.task_name
    exp_name = os.path.join(env_name, args.id, f"seed_{args.seed}")
    out_dir = os.path.join(args.out_dir, exp_name)

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

    # L = Logger(out_dir, use_tb=False)
    # evaluate(
    #     env=env,
    #     agent=agent,
    #     video=None,
    #     num_episodes=9,
    #     L=L,
    #     step=0,
    #     args=args,
    #     work_dir=args.out_dir,
    # )
    done = True

    for _ in range(args.num_samples):
        if done:
            obs = env.reset()
            done = False

        with eval_mode(agent):
            sample = np.random.random_sample()
            if sample > args.sample_percent:
                action = agent.select_action(obs / 255.0)
            else:
                action = agent.sample_action(obs / 255.0)

        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(obs, action, reward, next_obs, done)

    obses = replay_buffer.obses[: replay_buffer.idx]
    obses = agent.reshape_obses_for_actor(obs=obses)

    relu = torch.nn.ReLU()

    with torch.no_grad():
        out_enc = agent.actor.encoder(obses)
        out_trunk_LL_1 = agent.actor.trunk[0](out_enc)
        out_trunk_LL_2 = agent.actor.trunk[2](relu(out_trunk_LL_1))
        out_trunk_LL_3 = agent.actor.trunk[4](relu(out_trunk_LL_2))
        actions = relu(out_trunk_LL_3)

    fig, axes = plt.subplots(1, 5, figsize=(15, 5), sharey=True)
    fig.suptitle("t-SNE")

    # labels = np.zeros(actions.shape[0])

    # for arr in actions.unique(dim=0):
    #     mask = (actions == arr).all(axis=1)
    #     print('')

    create_PCA_subplot(z_tensor=out_enc, index=0, title="z encoded", axes=axes)
    create_PCA_subplot(
        z_tensor=out_trunk_LL_1, index=1, title="z trunk LL 1", axes=axes
    )
    create_PCA_subplot(
        z_tensor=out_trunk_LL_2, index=2, title="z trunk LL 2", axes=axes
    )
    create_PCA_subplot(
        z_tensor=out_trunk_LL_3, index=3, title="z trunk LL 3", axes=axes
    )
    create_PCA_subplot(z_tensor=actions, index=4, title="actions", axes=axes)

    plt.savefig("test.png")


if __name__ == "__main__":
    main()
