import json
import torch
import numpy as np
import dmc2gym
import gym
import os
import data_augs as rad
from collections import deque
import random
from torch.utils.data import Dataset
from curl_sac import RadSacAgent
from sklearn.metrics.pairwise import cosine_similarity
from data_augs import random_crop, random_translate


def set_json_to_args(args, config_path):
    config_dict = json.load(open(config_path))
    for key, value in config_dict.items():
        args.__dict__[key] = value

    return args


def create_env_and_replay_buffer(args, device):
    pre_transform_image_size = args.pre_transform_image_size
    pre_image_size = (
        args.pre_transform_image_size
    )  # record the pre transform image size for translation

    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=(args.encoder_type == "pixel"),
        height=pre_transform_image_size,
        width=pre_transform_image_size,
        frame_skip=args.action_repeat,
    )

    env.seed(args.seed)

    # stack several consecutive frames together
    if args.encoder_type == "pixel":
        env = FrameStack(env, k=args.frame_stack)

        action_shape = env.action_space.shape

    if args.encoder_type == "pixel":
        obs_shape = (3 * args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (
            3 * args.frame_stack,
            pre_transform_image_size,
            pre_transform_image_size,
        )
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    replay_buffer = ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
        pre_image_size=pre_image_size,
    )

    return env, replay_buffer


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == "rad_sac":
        return RadSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            latent_dim=args.latent_dim,
            data_augs=args.data_augs,
            mode=args.mode,
        )
    else:
        assert "agent is not supported: %s" % args.agent


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2 ** (8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        capacity,
        batch_size,
        device,
        image_size=84,
        pre_image_size=84,
        cpc_batch_size=None,
        transform=None,
    ):
        if cpc_batch_size is None:
            self.cpc_batch_size = batch_size
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.pre_image_size = pre_image_size  # for translation
        self.transform = transform
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.obses_other_env = np.empty((capacity, *obs_shape), dtype=obs_dtype)

        self.idx_other_env = 0
        self.idx = 0
        self.last_save = 0
        self.full = False
        self.full_other_env = False

        self.other_env = dmc2gym.make(
            domain_name="hopper",
            task_name="hop",
            seed=0,
            visualize_reward=False,
            from_pixels=True,
            height=self.pre_image_size,
            width=self.pre_image_size,
            frame_skip=2,
        )

        self.other_env = FrameStack(self.other_env, int(obs_shape[0] / 3))
        self.obses_other_env[self.idx_other_env] = self.other_env.reset()
        self.idx_other_env += 1

    def add(self, obs, action, reward, next_obs, done):

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def step_other_env(self):
        obs, _, done, _ = self.other_env.step(self.other_env.action_space.sample())
        self.obses_other_env[self.idx_other_env] = obs
        self.idx_other_env = (self.idx_other_env + 1) % self.capacity
        self.full_other_env = self.full_other_env or self.idx_other_env == 0

        if done:
            obs = self.other_env.reset()

    def sample_other_env(self):
        if not self.full_other_env or self.idx_other_env < self.batch_size:
            while self.idx_other_env < self.batch_size:
                self.step_other_env()
        else:
            self.step_other_env()

        idxs = np.random.randint(
            0,
            self.capacity if self.full_other_env else self.idx_other_env,
            size=self.batch_size,
        )

        return self.obses_other_env[idxs]

    def sample_proprio(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones

    def sample_contrastive(self, use_translate=False, use_other_env=False):
        max_buffer_sz = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, max_buffer_sz, size=self.cpc_batch_size)
        obses_contrastive = self.obses

        cpc_idxs = np.random.randint(
            0, len(obses_contrastive), size=self.cpc_batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        anchor = obses_contrastive[cpc_idxs]
        pos = anchor.copy()

        if use_translate:
            obses, translate_idxs = random_translate(
                imgs=obses, size=self.image_size, return_random_idxs=True
            )
            next_obses = random_translate(
                imgs=next_obses, size=self.image_size, **translate_idxs
            )
            anchor = random_translate(imgs=anchor, size=self.image_size)
            pos = random_translate(imgs=pos, size=self.image_size)
        else:
            obses = random_crop(obses, self.image_size)
            next_obses = random_crop(next_obses, self.image_size)
            pos = random_crop(pos, self.image_size)
            anchor = random_crop(anchor, self.image_size)

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        anchor = torch.as_tensor(anchor, device=self.device).float()
        pos = torch.as_tensor(pos, device=self.device).float()

        if use_other_env:
            obs_other_env = self.sample_other_env()

            if use_translate:
                obs_other_env = random_translate(
                    imgs=obs_other_env, size=self.image_size
                )
            else:
                obs_other_env = random_crop(obs_other_env, self.image_size)
            obs_other_env = torch.as_tensor(obs_other_env, device=self.device).float()
        else:
            obs_other_env = None

        contrastive_kwargs = dict(
            obs_anchor=anchor,
            obs_pos=pos,
            time_anchor=None,
            time_pos=None,
            obs_other_env=obs_other_env,
        )

        return obses, actions, rewards, next_obses, not_dones, contrastive_kwargs

    def sample_cluster(self, aug_funcs, use_translate=False, anchor_aug_funcs=None):
        # augs specified as flags
        # curl_sac organizes flags into aug funcs
        # passes aug funcs into sampler
        aug_obs_list = list()
        max_buffer_sz = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, max_buffer_sz, size=self.batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        pos = obses.copy()

        if use_translate:
            obses, translate_idxs = random_translate(
                imgs=obses, size=self.image_size, return_random_idxs=True
            )
            pos = random_translate(imgs=pos, size=self.image_size)
            next_obses = random_translate(
                imgs=next_obses, size=self.image_size, **translate_idxs
            )
        else:
            obses = random_crop(obses, self.image_size)
            pos = random_crop(imgs=pos, out=self.image_size)
            # next_obses = random_crop(next_obses, self.image_size)
            next_obses = center_crop_images(next_obses, output_size=self.image_size)

        obses = torch.as_tensor(obses, device=self.device).float()
        pos = torch.as_tensor(pos, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        obses = obses / 255.0
        next_obses = next_obses / 255.0
        # TODO: pos was not normalized before and got 300 reward on random conv with
        # super learner
        pos = pos / 255.0

        aug_obs_list.append(pos)

        # augmentations go here
        if aug_funcs:
            for aug, func_dict in aug_funcs.items():
                func = func_dict["func"]
                params = func_dict["params"]
                # skip crop and cutout augs

                if "crop" in aug or "cutout" in aug or "translate" in aug:
                    continue
                elif "instdisc" in aug or "set2" in aug:
                    params["return_all"] = True
                    aug_obs_list += func(pos, **params)
                else:
                    aug_obs_list.append(func(pos, **params))

        return obses, actions, rewards, next_obses, not_dones, aug_obs_list

    # def sample_super_learner(self, aug_funcs, use_translate=False):
    #     # augs specified as flags
    #     # curl_sac organizes flags into aug funcs
    #     # passes aug funcs into sampler
    #     aug_obs_list = list()
    #     max_buffer_sz = self.capacity if self.full else self.idx
    #     idxs = np.random.randint(0, max_buffer_sz, size=self.batch_size)

    #     obses = self.obses[idxs]
    #     pos = obses.copy()

    #     if use_translate:
    #         obses = center_translates(imgs=obses, size=self.image_size)
    #         pos = center_translates(imgs=pos, size=self.image_size)
    #     else:
    #         obses = center_crop_images(obses, output_size=self.image_size)
    #         pos = center_crop_images(pos, output_size=self.image_size)

    #     obses = torch.as_tensor(obses, device=self.device).float()
    #     pos = torch.as_tensor(pos, device=self.device).float()
    #     actions = torch.as_tensor(self.actions[idxs], device=self.device)
    #     rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
    #     not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

    #     obses = obses / 255.0
    #     next_obses = next_obses / 255.0

    #     aug_obs_list.append(pos)

    #     # augmentations go here
    #     if aug_funcs:
    #         for aug, func_dict in aug_funcs.items():
    #             func = func_dict["func"]
    #             params = func_dict["params"]
    #             # skip crop and cutout augs

    #             if "crop" in aug or "cutout" in aug or "translate" in aug:
    #                 continue
    #             elif "instdisc" in aug or "set2" in aug:
    #                 params["return_all"] = True
    #                 aug_obs_list += func(pos, **params)
    #             else:
    #                 aug_obs_list.append(func(pos, **params))

    #     return obses, actions, rewards, next_obses, not_dones, aug_obs_list

    def sample_rad(
        self,
        aug_funcs,
        idxs=None,
        return_idxs=False,
        aug_obs_only=False,
        aug_next_only=False,
    ):
        # augs specified as flags
        # curl_sac organizes flags into aug funcs
        # passes aug funcs into sampler

        if idxs is None:
            idxs = np.random.randint(
                0, self.capacity if self.full else self.idx, size=self.batch_size
            )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        if aug_funcs:
            for aug, func_dict in aug_funcs.items():
                func = func_dict["func"]
                params = func_dict["params"]
                # apply crop and cutout first
                if "crop" in aug or "cutout" in aug:
                    obses = func(obses, **params)
                    next_obses = func(next_obses, **params)
                elif "translate" in aug:
                    og_obses = center_crop_images(obses, self.pre_image_size)
                    obses, rndm_idxs = func(
                        og_obses, self.image_size, return_random_idxs=True
                    )

                    og_next_obses = center_crop_images(next_obses, self.pre_image_size)
                    next_obses = func(og_next_obses, self.image_size, **rndm_idxs)

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        obses = obses / 255.0
        next_obses = next_obses / 255.0

        # augmentations go here
        if aug_funcs:
            for aug, func_dict in aug_funcs.items():
                func = func_dict["func"]
                params = func_dict["params"]
                # skip crop and cutout augs

                if "crop" in aug or "cutout" in aug or "translate" in aug:
                    continue

                obses = func(obses, **params)
                next_obses = func(next_obses, **params)

        if aug_next_only:
            obses = self.obses[idxs]

            if "crop" in aug_funcs:
                obses = center_crop_images(obses, output_size=self.image_size)
            elif "translate" in aug_funcs:
                og_obses = center_crop_images(obses, self.pre_image_size)
                obses = center_translates(og_obses, size=self.image_size)

            obses = torch.as_tensor(obses, device=self.device) / 255.0
        elif aug_obs_only:
            next_obses = self.next_obses[idxs]

            if "crop" in aug_funcs:
                next_obses = center_crop_images(next_obses, output_size=self.image_size)
            elif "translate" in aug_funcs:
                og_next_obses = center_crop_images(next_obses, self.pre_image_size)
                next_obses = center_translates(og_next_obses, size=self.image_size)

            next_obses = torch.as_tensor(next_obses, device=self.device) / 255.0

        if return_idxs:
            return obses, actions, rewards, next_obses, not_dones, idxs
        else:
            return obses, actions, rewards, next_obses, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, "%d_%d.pt" % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save : self.idx],
            self.next_obses[self.last_save : self.idx],
            self.actions[self.last_save : self.idx],
            self.rewards[self.last_save : self.idx],
            self.not_dones[self.last_save : self.idx],
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split("_")[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split(".")[0].split("_")]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

    def __getitem__(self, idx):
        idx = np.random.randint(0, self.capacity if self.full else self.idx, size=1)
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype,
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    image = image[:, top : top + new_h, left : left + new_w]
    return image


def center_crop_images(image, output_size):
    h, w = image.shape[2:]
    new_h, new_w = output_size, output_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    image = image[:, :, top : top + new_h, left : left + new_w]
    return image


def center_translate(image, size):
    c, h, w = image.shape
    assert size >= h and size >= w
    outs = np.zeros((c, size, size), dtype=image.dtype)
    h1 = (size - h) // 2
    w1 = (size - w) // 2
    outs[:, h1 : h1 + h, w1 : w1 + w] = image
    return outs


def center_translates(image, size):
    b, c, h, w = image.shape
    assert size >= h and size >= w
    outs = np.zeros((b, c, size, size), dtype=image.dtype)
    h1 = (size - h) // 2
    w1 = (size - w) // 2
    outs[:, :, h1 : h1 + h, w1 : w1 + w] = image
    return outs
