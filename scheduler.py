import numpy as np
import gymnasium as gym

from ribs.archives import CVTArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

from models import MLP

def create_scheduler(seed, n_emitters, sigma0, batch_size):
    """Creates the Scheduler based on given configurations.

    See lunar_lander_main() for description of args.

    Returns:
        A pyribs scheduler set up for CMA-ME (i.e. it has
        EvolutionStrategyEmitter's and a GridArchive).
"""
    env = gym.make("LunarLander-v2")
    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    initial_model = MLP(obs_dim, action_dim, 64, 64)

    # archive = GridArchive(
    #     solution_dim=initial_model.size,
    #     dims=[50, 50],  # 50 cells in each dimension.
    #     # (-1, 1) for x-pos and (-3, 0) for y-vel.
    #     ranges=[(-1.0, 1.0), (-3.0, 0.0)],
    #     qd_score_offset=-600,
    #     seed=seed)
    
    archive = CVTArchive(
        solution_dim=sum(p.numel() for p in initial_model.parameters() if p.requires_grad), 
        cells=1000, 
        ranges=[(-1.0, 1.0), (-3.0, 0.0)], 
        learning_rate=1.0,
        qd_score_offset=-600, 
        seed=seed,  
        #samples=100000, 
        custom_centroids=None, 
        k_means_kwargs=None, 
        use_kd_tree=True, 
        ckdtree_kwargs=None)

    
    # If we create the emitters with identical seeds, they will all output the
    # same initial solutions. The algorithm should still work -- eventually, the
    # emitters will produce different solutions because they get different
    # responses when inserting into the archive. However, using different seeds
    # avoids this problem altogether.
    seeds = ([None] * n_emitters
             if seed is None else [seed + i for i in range(n_emitters)])

    # We use the EvolutionStrategyEmitter to create an ImprovementEmitter.
    emitters = [
        EvolutionStrategyEmitter(
            archive,
            #x0=initial_model.flatten(),
            #x0=np.concatenate([p.data.cpu().detach().numpy().ravel() for p in initial_model.parameters()]),
            x0=initial_model.serialize(),
            sigma0=sigma0,
            ranker="2imp",
            batch_size=batch_size,
            seed=s,
        ) for s in seeds
    ]

    scheduler = Scheduler(archive, emitters)
    return scheduler