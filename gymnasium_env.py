import numpy as np
import gymnasium as gym


class GymnasiumEnv():

    def __init__( \
        self, env_id = "CartPole-v1",
        render_mode = "rgb_array", n_max_step = 1000,
        sutton_barto_reward = True) -> None:

        self.env = gym.make(
            env_id,
            render_mode=render_mode,
            max_episode_steps=n_max_step,
            sutton_barto_reward=sutton_barto_reward,
        )
        self.terminated = False
        self.truncated = False


    @property
    def ended(self):

        return self.terminated | self.truncated


    def step(self, action):

        observation, reward, self.terminated, self.truncated, _ = self.env.step(action)
        return observation, reward


    def close(self):
        self.env.close()
