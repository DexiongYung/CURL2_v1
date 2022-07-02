import os
import time
import utils
import numpy as np


def run_eval(env, agent, video, video_enabled, args, sample_stochastically=False):
    obs = env.reset()
    if video is not None:
        video.init(enabled=(video_enabled))
    done = False
    episode_reward = 0
    while not done:
        # center crop image
        if args.encoder_type == "pixel" and "crop" in args.data_augs:
            obs = utils.center_crop_image(obs, args.image_size)
        if args.encoder_type == "pixel" and "translate" in args.data_augs:
            # first crop the center with pre_image_size
            obs = utils.center_crop_image(obs, args.pre_transform_image_size)
            # then translate cropped to center
            obs = utils.center_translate(obs, args.image_size)
        with utils.eval_mode(agent):
            if sample_stochastically:
                action = agent.sample_action(obs / 255.0)
            else:
                action = agent.select_action(obs / 255.0)
            obs, reward, done, _ = env.step(action)

            if video is not None:
                video.record(env)
            episode_reward += reward

    return episode_reward


def evaluate(env, agent, video, num_episodes, L, step, args, work_dir):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = "stochastic_" if sample_stochastically else ""
        for i in range(num_episodes):
            episode_reward = run_eval(
                env=env,
                agent=agent,
                video=video,
                video_enabled=i == 1,
                args=args,
                sample_stochastically=sample_stochastically,
            )

            if video is not None:
                video.save("%d.mp4" % step)
            L.log("eval/" + prefix + "episode_reward", episode_reward, step)
            all_ep_rewards.append(episode_reward)

        L.log("eval/" + prefix + "eval_time", time.time() - start_time, step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        std_ep_reward = np.std(all_ep_rewards)
        L.log("eval/" + prefix + "mean_episode_reward", mean_ep_reward, step)
        L.log("eval/" + prefix + "best_episode_reward", best_ep_reward, step)

        filename = os.path.join(work_dir, "eval_scores.npy")
        key = args.domain_name + "-" + args.task_name + "-" + args.data_augs
        try:
            log_data = np.load(filename, allow_pickle=True)
            log_data = log_data.item()
        except:
            log_data = {}

        if key not in log_data:
            log_data[key] = {}

        log_data[key][step] = {}
        log_data[key][step]["step"] = step
        log_data[key][step]["mean_ep_reward"] = mean_ep_reward
        log_data[key][step]["max_ep_reward"] = best_ep_reward
        log_data[key][step]["std_ep_reward"] = std_ep_reward
        log_data[key][step]["env_step"] = step * args.action_repeat

        np.save(filename, log_data)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)
