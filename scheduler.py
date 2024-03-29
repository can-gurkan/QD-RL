import numpy as np
import gymnasium as gym
import gin

from ribs.schedulers import Scheduler
from models import MLP


@gin.configurable
def create_scheduler(seed, n_emitters, sigma0, batch_size, archive_type=gin.REQUIRED, emitter_type=gin.REQUIRED, env_name=gin.REQUIRED):
    """Creates the Scheduler based on given configurations.

    Returns:
        A pyribs scheduler set up for an ME-variant.
"""
    env = gym.make(env_name)
    if env_name == "LunarLander-v2":
        action_dim = env.action_space.n
        obs_dim = env.observation_space.shape[0]
    elif env_name == "HalfCheetah-v4":
        action_dim = env.action_space.shape
        obs_dim = env.observation_space.shape
    elif "MiniGrid" in env_name:
        #Fix this later to be able to accept any minigrid env
        action_dim = 7
        obs_dim = 49

    initial_model = MLP(obs_dim, action_dim)
    archive = archive_type(
        solution_dim=len(initial_model.serialize())
    )

    # If we create the emitters with identical seeds, they will all output the
    # same initial solutions. The algorithm should still work -- eventually, the
    # emitters will produce different solutions because they get different
    # responses when inserting into the archive. However, using different seeds
    # avoids this problem altogether.
    seeds = ([None] * n_emitters
             if seed is None else [seed + i for i in range(n_emitters)])

    emitters = [
        emitter_type(
            archive,
            x0=initial_model.serialize(),
            sigma0=sigma0,
            batch_size=batch_size,
            seed=s,
        ) for s in seeds
    ]

    scheduler = Scheduler(archive, emitters)
    return scheduler