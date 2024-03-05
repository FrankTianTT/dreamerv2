from typing import Optional

import minatar
import gym
import numpy as np
import os
from matplotlib import colors
import seaborn as sns
from scipy import ndimage
from dreamerv2.utils.imgsource import RandomVideoSource, RandomImageSource

pomdp_index = {
    'breakout': [0, 1, 3],
    'seaquest': [0, 1, 2, 4, 5, 6, 7, 8, 9],
    'space_invaders': [0, 1, 4, 5],
    'asterix': [0, 1, 3],
    'freeway': [0, 1],
}


class GymMinAtar(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
            self,
            env_name,
            display_time=50,
            obs_type="pixel",
            noise_alpha=0.3,
            pixel_size=40,
            noise_type=None,
            resource_dir=None
    ):
        self.display_time = display_time
        self.env_name = env_name
        self.env = minatar.Environment(env_name)
        # obs_type: 'pixel', 'mdp' or 'pomdp'
        self.obs_type = obs_type
        self.noise_alpha = noise_alpha
        self.pixel_size = pixel_size
        self.noise_type = noise_type
        self.resource_dir = resource_dir

        self.minimal_actions = self.env.minimal_action_set()
        self.action_space = gym.spaces.Discrete(len(self.minimal_actions))

        h, w, c = self.env.state_shape()
        if obs_type == "pixel":
            self.observation_space = gym.spaces.Box(0, 1, (3, self.pixel_size, self.pixel_size))
            self.cmap = sns.color_palette("cubehelix", self.env.n_channels)
            self.cmap.insert(0, (0, 0, 0))
            self.cmap = colors.ListedColormap(self.cmap)
        elif obs_type == "mdp":
            self.observation_space = gym.spaces.MultiBinary((c, h, w))
        elif obs_type == "pomdp":
            self.observation_space = gym.spaces.MultiBinary((len(pomdp_index[env_name]), h, w))

        if self.noise_type is None or self.noise_alpha == 0:
            self.img_source, self.resource_files = None, None
        elif self.noise_type == "videos":
            self.resource_files = [os.path.join(resource_dir, f)
                                   for f in os.listdir(self.resource_dir)
                                   if f.endswith('.mp4')]
            self.img_source = RandomVideoSource((self.pixel_size, self.pixel_size), self.resource_files)
        elif self.noise_type == "images":
            self.resource_files = [os.path.join(resource_dir, f)
                                   for f in os.listdir(self.resource_dir)
                                   if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]
            self.img_source = RandomImageSource((self.pixel_size, self.pixel_size), self.resource_files)
        else:
            raise ValueError("Invalid noise source")

    def get_obs(self):
        state = self.env.state()
        if self.obs_type == "pixel":
            rgb_array = self.render('rgb_array')
            return rgb_array.transpose(2, 0, 1)
        elif self.obs_type == "mdp":
            return state.transpose(2, 0, 1)
        elif self.obs_type == "pomdp":
            obs = []
            for i in pomdp_index[self.env_name]:
                obs.append(state[:, :, i])
            return np.stack(obs, axis=0)

    def reset(self, seed: Optional[int] = None, **kwargs):
        self.env = minatar.Environment(self.env_name)
        self.env.seed(seed)
        if self.img_source:
            self.img_source.reset()
        return self.get_obs(), {}

    def step(self, index):
        '''index is the action id, considering only the set of minimal actions'''
        action = self.minimal_actions[index]
        r, terminal = self.env.act(action)
        self.game_over = terminal
        return self.get_obs(), r, terminal, False, {}

    def render(self, mode='rgb_array'):
        assert mode == 'rgb_array', 'Only support rgb_array mode'

        state = self.env.state()

        numerical_state = np.amax(state * np.reshape(np.arange(self.env.n_channels) + 1, (1, 1, -1)), 2)
        rgb_array = self.cmap(numerical_state)[..., :3]  # shape: (10, 10, 3)
        rgb_array = ndimage.zoom(
            rgb_array,
            zoom=(self.pixel_size / rgb_array.shape[0], self.pixel_size / rgb_array.shape[1], 1),
            mode="nearest",
            order=0
        )
        if self.img_source:
            mask = (rgb_array == (0, 0, 0)).astype(float) * self.noise_alpha
            img = self.img_source.get_image()
            rgb_array[rgb_array == (0, 0, 0)] = [1, 1, 1]
            rgb_array = rgb_array * (1 - mask) + img * mask
        return rgb_array.astype(np.float32)

    def close(self):
        if self.env.visualized:
            self.env.close_display()
        return 0


class breakoutPOMDP(gym.ObservationWrapper):
    def __init__(self, env):
        '''index 2 (trail) is removed, which gives ball's direction'''
        super(breakoutPOMDP, self).__init__(env)
        c, h, w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((c - 1, h, w))

    def observation(self, observation):
        return np.stack([observation[0], observation[1], observation[3]], axis=0)


class ActionRepeat(gym.Wrapper):
    def __init__(self, env, repeat=1):
        super(ActionRepeat, self).__init__(env)
        self.repeat = repeat

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self.repeat and not done:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            current_step += 1
        return obs, total_reward, done, info


class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        super(TimeLimit, self).__init__(env)
        self._duration = duration
        self._step = 0

    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            info['time_limit_reached'] = True
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self.env.reset()


class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete), "This wrapper only works with discrete action space"
        shape = (env.action_space.n,)
        env.action_space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        env.action_space.sample = self._sample_action
        super(OneHotAction, self).__init__(env)

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        return self.env.step(index)

    def reset(self):
        return self.env.reset()

    def _sample_action(self):
        actions = self.env.action_space.shape[0]
        index = np.random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference
