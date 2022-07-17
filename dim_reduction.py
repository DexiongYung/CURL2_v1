import os
from statistics import mean
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import torch
import argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
from numpy import reshape
from logger import Logger
from seaborn import scatterplot
from evaluation.model_loading import load_actor
from evaluation.evaluate import evaluate
from utils import make_agent, set_json_to_args, create_env_and_replay_buffer, eval_mode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model_step", type=int, default=99999)
    parser.add_argument("--total_samples", type=int, default=1000)
    parser.add_argument("--num_random_samples", type=int, default=0)
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
    z_enc = tsne.fit_transform(z_tensor)
    axes[index].scatter(x=z_enc[:, 0], y=z_enc[:, 1])
    axes[index].set_title(title)


def create_labeled_t_SNE(z_tensor, labels, axes, index: int):
    tsne = TSNE(n_components=2, random_state=123)
    z_enc = tsne.fit_transform(z_tensor)
    full_arr = np.concatenate((z_enc, labels[:, None]), axis=1)
    df = pd.DataFrame(full_arr, columns=["x", "y", "label"])
    df["label"] = df.label.astype("category")
    scatterplot(data=df, x="x", y="y", hue="label", ax=axes[index])


def k_means_w_silhouette(z, kmax):
    best_sil = float("-inf")
    best_labels = None

    for k in range(2, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(z)
        labels = kmeans.labels_
        curr_sil = silhouette_score(z, labels, metric="euclidean")

        if curr_sil > best_sil:
            best_sil = curr_sil
            best_labels = labels

    return best_labels, best_sil


def GMM_w_silhouette(z, kmax):
    best_sil = float("-inf")
    best_labels = None

    for k in range(2, kmax + 1):
        gmm = GaussianMixture(n_components=k).fit(z)
        labels = gmm.predict(z)
        curr_sil = silhouette_score(z, labels, metric="euclidean")

        if curr_sil > best_sil:
            best_sil = curr_sil
            best_labels = labels

    return best_labels, best_sil


def dbscan_w_silhouette(z, eps):
    dbscan = DBSCAN(eps=eps).fit(z)
    labels = dbscan.labels_
    sil_score = silhouette_score(z, labels, metric="euclidean")

    return labels, sil_score


def get_encoding_each_layer(agent, obses):
    relu = torch.nn.ReLU()

    with torch.no_grad():
        out_enc = agent.actor.encoder(obses)
        out_trunk_LL_1 = agent.actor.trunk[0](out_enc)
        out_trunk_LL_2 = agent.actor.trunk[2](relu(out_trunk_LL_1))
        out_trunk_LL_3 = agent.actor.trunk[4](relu(out_trunk_LL_2))
        actions = relu(out_trunk_LL_3)

    return out_enc, out_trunk_LL_1, out_trunk_LL_2, out_trunk_LL_3, actions


def create_subplots(size: int, title: str):
    fig, axes = plt.subplots(1, size, figsize=(15, 5), sharey=True)
    fig.suptitle(title)

    return fig, axes


def run_actor(agent, obses):
    relu = torch.nn.ReLU()

    with torch.no_grad():
        out_enc = agent.actor.encoder(obses).detach().cpu().numpy()
        # out_trunk_LL_1 = agent.actor.trunk[0](out_enc)
        # out_trunk_LL_2 = agent.actor.trunk[2](relu(out_trunk_LL_1))
        # out_trunk_LL_3 = agent.actor.trunk[4](relu(out_trunk_LL_2))
        # actions = relu(out_trunk_LL_3)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    fig.suptitle("t-SNE")

    # mu_actions, log_std_actions = actions.chunk(2, dim=-1)
    # mu_actions = mu_actions.detach().cpu().numpy()
    # log_std_actions = log_std_actions.detach().cpu().numpy()
    # radius = log_std_actions.mean() * 2

    # create_t_SNE_subplot(z_tensor=out_enc, index=0, title="z encoded", axes=axes)
    # create_t_SNE_subplot(
    #     z_tensor=out_trunk_LL_1, index=1, title="z trunk LL 1", axes=axes
    # )
    # create_t_SNE_subplot(
    #     z_tensor=out_trunk_LL_2, index=2, title="z trunk LL 2", axes=axes
    # )
    # create_t_SNE_subplot(
    #     z_tensor=out_trunk_LL_3, index=3, title="z trunk LL 3", axes=axes
    # )
    # create_t_SNE_subplot(z_tensor=actions, index=4, title="actions", axes=axes)

    # create_labeled_t_SNE(z_tensor=mu_actions, labels=best_labels, axes=axes, index=0)
    # create_labeled_t_SNE(
    #     z_tensor=out_enc.detach().cpu().numpy(), labels=best_labels, axes=axes, index=1
    # )

    # labels, sil_score = dbscan_w_silhouette(z=mu_actions, eps=radius)
    import time

    start_time = time.time()
    labels, sil_score = k_means_w_silhouette(z=out_enc, kmax=50)
    print("--- %s seconds ---" % (time.time() - start_time))

    plt.savefig(f"k_means.png")


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
    agent.load(model_dir=args.model_path, step=args.model_step)

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

    for step in range(args.total_samples):
        if done:
            obs = env.reset()
            done = False

        with eval_mode(agent):
            if step > args.num_random_samples:
                action = agent.select_action(obs / 255.0)
            else:
                action = env.action_space.sample()

        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs

    obses = replay_buffer.obses[: replay_buffer.idx]
    obses = np.unique(obses, axis=0)
    obses = agent.reshape_obses_for_actor(obs=obses)


if __name__ == "__main__":
    main()
