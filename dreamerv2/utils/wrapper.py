from typing import Optional

import minatar
import gym
import numpy as np

pomdp_index = {
    'breakout': [0, 1, 3],
    'seaquest': [0, 1, 2, 4, 5, 6, 7, 8, 9],
    'space_invaders': [0, 1, 4, 5],
    'asterix': [0, 1, 3],
    'freeway': [0, 1],
}


class GymMinAtar(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, env_name, display_time=50, obs_type="pixel"):
        self.display_time = display_time
        self.env_name = env_name
        self.env = minatar.Environment(env_name)
        # obs_type: 'pixel', 'mdp' or 'pomdp'
        self.obs_type = obs_type

        self.minimal_actions = self.env.minimal_action_set()
        self.action_space = gym.spaces.Discrete(len(self.minimal_actions))

        h, w, c = self.env.state_shape()
        if obs_type == "pixel":
            from matplotlib import colors
            import seaborn as sns

            self.observation_space = gym.spaces.Box(0, 1, (3, h, w))
            self.cmap = sns.color_palette("cubehelix", self.env.n_channels)
            self.cmap.insert(0, (0, 0, 0))
            self.cmap = colors.ListedColormap(self.cmap)
        elif obs_type == "mdp":
            self.observation_space = gym.spaces.MultiBinary((c, h, w))
        elif obs_type == "pomdp":
            self.observation_space = gym.spaces.MultiBinary((len(pomdp_index[env_name]), h, w))

    def get_obs(self):
        state = self.env.state()
        if self.obs_type == "pixel":
            rgb_array = self.render('rgb_array')
            return rgb_array.transpose(2, 0, 1) - 0.5
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
        return self.cmap(numerical_state)[..., :3]

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
