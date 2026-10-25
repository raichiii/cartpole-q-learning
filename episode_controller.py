from gymnasium.wrappers import RecordVideo


class EpisodeController:
    def __init__(self, env, agent, n_max_step) -> None:
        self.env = env
        self.agent = agent
        self.n_max_step = n_max_step
        self.frames = []
        self.i_episode = 0

    def run(self, record = False):
        if record:
            self.env.env = RecordVideo(self.env.env, video_folder="video")

        self.frames = []
        observation, _ = self.env.env.reset(seed=self.i_episode)

        for step in range(self.n_max_step):

            if record:
                # Stock frames for gif
                self.frames.append(self.env.env.render())

            # Get action
            action = self.agent.get_action(observation, self.i_episode)

            observation_next, reward = self.env.step(action)
            self.agent.update_Q_function(observation, action, reward, observation_next)

            if self.env.ended:
                break

            observation = observation_next

        self.i_episode += 1

        return step
