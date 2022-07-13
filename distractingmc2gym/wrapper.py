import numpy as np
import os
import gym
import torch
import torch.nn.functional as F
import dmc2gym
import utils
from collections import deque


class LazyFrames(object):
    def __init__(self, frames, extremely_lazy=True):
        self._frames = frames
        self._extremely_lazy = extremely_lazy
        self._out = None

    @property
    def frames(self):
        return self._frames

    def _force(self):
        if self._extremely_lazy:
            return np.concatenate(self._frames, axis=0)
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        if self._extremely_lazy:
            return len(self._frames)
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        if self.extremely_lazy:
            return len(self._frames)
        frames = self._force()
        return frames.shape[0] // 3

    def frame(self, i):
        return self._force()[i * 3 : (i + 1) * 3]


class FrameStack(gym.Wrapper):
    """Stack frames as observation"""

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
        return LazyFrames(list(self._frames))


class ColorWrapper(gym.Wrapper):
    """Wrapper for the color experiments"""

    def __init__(self, env, mode, seed=None):
        assert isinstance(env, utils.FrameStack), "wrapped env must be a framestack"
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        self._mode = mode
        self._random_state = np.random.RandomState(seed)
        self.time_step = 0
        if "color" in self._mode:
            self._load_colors()

    def reset(self):
        self.time_step = 0
        if "color" in self._mode:
            self.randomize()
        elif "video" in self._mode:
            # apply greenscreen
            setting_kwargs = {
                "skybox_rgb": [0.2, 0.8, 0.2],
                "skybox_rgb2": [0.2, 0.8, 0.2],
                "skybox_markrgb": [0.2, 0.8, 0.2],
            }
            if self._mode == "video_hard":
                setting_kwargs["grid_rgb1"] = [0.2, 0.8, 0.2]
                setting_kwargs["grid_rgb2"] = [0.2, 0.8, 0.2]
                setting_kwargs["grid_markrgb"] = [0.2, 0.8, 0.2]
            self.reload_physics(setting_kwargs)
        return self.env.reset()

    def step(self, action):
        self.time_step += 1
        return self.env.step(action)

    def randomize(self):
        assert (
            "color" in self._mode
        ), f"can only randomize in color mode, received {self._mode}"
        self.reload_physics(self.get_random_color())

    def _load_colors(self):
        assert self._mode in {"color_easy", "color_hard"}
        self._colors = torch.load(f"./distractingmc2gym/data/{self._mode}.pt")

    def get_random_color(self):
        assert len(self._colors) >= 100, "env must include at least 100 colors"
        return self._colors[self._random_state.randint(len(self._colors))]

    def reload_physics(self, setting_kwargs=None, state=None):
        from dm_control.suite import common

        domain_name = self._get_dmc_wrapper()._domain_name
        if setting_kwargs is None:
            setting_kwargs = {}
        if state is None:
            state = self._get_state()
        self._reload_physics(
            *common.settings.get_model_and_assets_from_setting_kwargs(
                domain_name + ".xml", setting_kwargs
            )
        )
        self._set_state(state)

    def get_state(self):
        return self._get_state()

    def set_state(self, state):
        self._set_state(state)

    def _get_dmc_wrapper(self):
        _env = self.env
        while not isinstance(_env, dmc2gym.wrappers.DMCWrapper) and hasattr(
            _env, "env"
        ):
            _env = _env.env
        assert isinstance(
            _env, dmc2gym.wrappers.DMCWrapper
        ), "environment is not dmc2gym-wrapped"

        return _env

    def _reload_physics(self, xml_string, assets=None):
        _env = self.env
        while not hasattr(_env, "_physics") and hasattr(_env, "env"):
            _env = _env.env
        assert hasattr(_env, "_physics"), "environment does not have physics attribute"
        _env.physics.reload_from_xml_string(xml_string, assets=assets)

    def _get_physics(self):
        _env = self.env
        while not hasattr(_env, "_physics") and hasattr(_env, "env"):
            _env = _env.env
        assert hasattr(_env, "_physics"), "environment does not have physics attribute"

        return _env._physics

    def _get_state(self):
        return self._get_physics().get_state()

    def _set_state(self, state):
        self._get_physics().set_state(state)


class VideoWrapper(gym.Wrapper):
    """Green screen for video experiments"""

    def __init__(self, env, mode, seed):
        gym.Wrapper.__init__(self, env)
        self._mode = mode
        self._seed = seed
        self._random_state = np.random.RandomState(seed)
        self._index = 0
        self._video_paths = []
        if "video" in mode:
            self._get_video_paths()
        self._num_videos = len(self._video_paths)
        self._max_episode_steps = env._max_episode_steps

    def _get_video_paths(self):
        video_dir = os.path.join("src/env/data", self._mode)
        if "video_easy" in self._mode:
            self._video_paths = [
                os.path.join(video_dir, f"video{i}.mp4") for i in range(10)
            ]
        elif "video_hard" in self._mode:
            self._video_paths = [
                os.path.join(video_dir, f"video{i}.mp4") for i in range(100)
            ]
        else:
            raise ValueError(f'received unknown mode "{self._mode}"')

    def _load_video(self, video):
        """Load video from provided filepath and return as numpy array"""
        import cv2

        cap = cv2.VideoCapture(video)
        assert (
            cap.get(cv2.CAP_PROP_FRAME_WIDTH) >= 100
        ), "width must be at least 100 pixels"
        assert (
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT) >= 100
        ), "height must be at least 100 pixels"
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        buf = np.empty(
            (
                n,
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                3,
            ),
            np.dtype("uint8"),
        )
        i, ret = 0, True
        while i < n and ret:
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf[i] = frame
            i += 1
        cap.release()
        return np.moveaxis(buf, -1, 1)

    def _reset_video(self):
        self._index = (self._index + 1) % self._num_videos
        self._data = self._load_video(self._video_paths[self._index])

    def reset(self):
        if "video" in self._mode:
            self._reset_video()
        self._current_frame = 0
        return self._greenscreen(self.env.reset())

    def step(self, action):
        self._current_frame += 1
        obs, reward, done, info = self.env.step(action)
        return self._greenscreen(obs), reward, done, info

    def _interpolate_bg(self, bg, size: tuple):
        """Interpolate background to size of observation"""
        bg = torch.from_numpy(bg).float().unsqueeze(0) / 255.0
        bg = F.interpolate(bg, size=size, mode="bilinear", align_corners=False)
        return (bg * 255.0).byte().squeeze(0).numpy()

    def _greenscreen(self, obs):
        """Applies greenscreen if video is selected, otherwise does nothing"""
        if "video" in self._mode:
            bg = self._data[self._current_frame % len(self._data)]  # select frame
            bg = self._interpolate_bg(bg, obs.shape[1:])  # scale bg to observation size
            return do_green_screen(obs, bg)  # apply greenscreen
        return obs

    def apply_to(self, obs):
        """Applies greenscreen mode of object to observation"""
        obs = obs.copy()
        channels_last = obs.shape[-1] == 3
        if channels_last:
            obs = torch.from_numpy(obs).permute(2, 0, 1).numpy()
        obs = self._greenscreen(obs)
        if channels_last:
            obs = torch.from_numpy(obs).permute(1, 2, 0).numpy()
        return obs
