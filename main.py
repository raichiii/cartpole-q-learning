from logging import getLogger
import hydra
from omegaconf import DictConfig

from gymnasium_env import GymnasiumEnv
from agent import Agent
from episode_controller import EpisodeController
from util import make_gif_from_ndarray


logger = getLogger(__name__)


@hydra.main(config_path="", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # Parameters
    env_id = cfg.env_id
    rl = cfg.rl
    gamma = cfg.gamma
    d_digitized = cfg.d_digitized
    sutton_barto_reward = cfg.sutton_barto_reward
    n_max_episode = cfg.n_max_episode
    n_max_step = cfg.n_max_step
    n_step_success = cfg.n_step_success
    n_max_continuous_success = cfg.n_max_continuous_success
    use_seed = cfg.reproducibility.use_seed
    fname_gif = cfg.display.fname_gif
    duration_gif = cfg.display.duration_gif
    n_episode_per_print = cfg.logging.n_episode_per_print

    # Confirm conditions
    assert n_step_success < n_max_step

    # Get instances
    env = GymnasiumEnv(env_id=env_id, n_max_step=n_max_step, sutton_barto_reward=sutton_barto_reward)
    agent = Agent(env=env, rl=rl, gamma=gamma, d_digitized=d_digitized, use_seed=use_seed)
    episode_ctrl = EpisodeController(env=env, agent=agent, n_max_step=n_max_step)
    n_continuous_success = 0
    record = False

    for i_episode in range(n_max_episode):

        # Run a episode
        step = episode_ctrl.run(record=record)

        if i_episode % n_episode_per_print == 0:
            logger.info(f"Episode {i_episode}: {step} steps")

        if n_step_success <= step:
            n_continuous_success += 1
        else:
            n_continuous_success = 0

        if record: break

        if n_max_continuous_success == n_continuous_success:
            logger.info("Succeeded 10 times")
            agent.eval()
            record = True

    if record:
        make_gif_from_ndarray(
            episode_ctrl.frames,
            fname_gif,
            duration=duration_gif,
        )

    env.close()


if __name__ == "__main__":
    main()
