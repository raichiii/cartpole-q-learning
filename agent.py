from typing import Type
import numpy as np
import gymnasium as gym

from gymnasium_env import GymnasiumEnv


def get_dimension_from_space(space):

    if isinstance(space, gym.spaces.Box):
        return space.shape[0]
    if isinstance(space, gym.spaces.Discrete):
        return space.n


class Agent():

    def __init__( \
        self, env: GymnasiumEnv,
        rl = 0.5, gamma = 0.99,
        d_digitized = 6, use_seed = True) -> None:

        self.env = env
        self.rl = rl
        self.gamma = gamma
        self.d_digitized = d_digitized
        self.d_observation = get_dimension_from_space(self.env.env.observation_space)
        self.d_action = get_dimension_from_space(self.env.env.action_space)
        self.rng = np.random.default_rng(seed=0) if use_seed else np.random
        self.Q_function = self.init_Q_function()
        self.bins_digitization = [
            self.create_bins_digitization(-2.4, 2.4, self.d_digitized),
            self.create_bins_digitization(-3.0, 2.4, self.d_digitized),
            self.create_bins_digitization(-0.5, 0.5, self.d_digitized),
            self.create_bins_digitization(-2.0, 2.0, self.d_digitized),
        ]
        self.train()

    def init_Q_function(self, high = 1):

        return self.rng.uniform(
            low=0, high=high, size=[self.d_digitized] * self.d_observation + [self.d_action],
        )

    def update_Q_function(self, observation, action, reward, observation_next):
        """Update Q function"""

        state = self.digitize_state(observation)
        state_next = self.digitize_state(observation_next)
        Max_Q_next = max(self.Q_function[*state_next]) # Get max action value on the next state

        # HERE is Q learning
        self.Q_function[*state, action] = self.Q_function[*state, action] + self.rl * (reward + self.gamma * Max_Q_next - self.Q_function[*state, action])

    def create_bins_digitization(self, clip_min, clip_max, num):
        """Get thresholds for transforming continuous into discrete
        """

        return np.linspace(clip_min, clip_max, num + 1)[1:-1]

    def digitize_state(self, observation):
        """Transform continuous values into discrete
        Example:
            x => [-3, 1, 2, 3, 4, 5]
            bins => [1.5, 2, 3.7]

            Because of -3, 1 < 1.5 < 2 = 2 < 3 < 3.7 < 4, 5
                       -----         -       -         ----
                         0           1       2          3

            np.digitize(x, bins) => [0, 1, 2, 3, 3]
        """

        return [
            np.digitize(obs, bins=b)
            for obs, b in zip(observation, self.bins_digitization)
        ]

    def __get_epsilon_greedy_action(self, observation, episode, *args, **kwargs):
        """Decide action with ε-greedy method
        """

        epsilon = 0.5 * (1 / (episode + 1)) # Epsilon decreases as the number of episodes increases

        if epsilon <= self.rng.uniform(0, 1):
            return self.__get_optimal_action(observation=observation)
        else:
            return self.rng.choice(self.d_action)

    def __get_optimal_action(self, observation, *args, **kwargs):
        """Get optimal action
        """

        state = self.digitize_state(observation)
        action = np.argmax(self.Q_function[*state])
        return action

    def train(self):
        self.get_action = self.__get_epsilon_greedy_action

    def eval(self):
        self.get_action = self.__get_optimal_action
