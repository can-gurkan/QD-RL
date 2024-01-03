import gymnasium as gym
import numpy as np
import gin
from models import MLP
from qd_gym import QDHalfCheetahWrapper


def simulate_LL(sol, seed=None, video_env=None):
    """Simulates the lunar lander environment.

    Args:
        model (np.ndarray): The array of weights for the linear policy.
        seed (int): The seed for the environment.
        video_env (gym.Env): If passed in, this will be used instead of creating
            a new env. This is used primarily for recording video during
            evaluation.
    Returns:
        total_reward (float): The reward accrued by the lander throughout its
            trajectory.
        impact_x_pos (float): The x position of the lander when it touches the
            ground for the first time.
        impact_y_vel (float): The y velocity of the lander when it touches the
            ground for the first time.
    """
    
    if video_env is None:
        env = gym.make("LunarLander-v2")
    else:
        env = video_env

    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    
    gin.parse_config_file('config/nnparams.gin')

    model =  MLP(obs_dim, action_dim).deserialize(sol)

    total_reward = 0.0
    impact_x_pos = None
    impact_y_vel = None
    all_y_vels = []
    obs, _ = env.reset(seed=seed)
    done = False

    while not done:
        action = model.choose_action_disc(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Refer to the definition of state here:
        # https://gymnasium.farama.org/environments/box2d/lunar_lander/
        x_pos = obs[0]
        y_vel = obs[3]
        leg0_touch = bool(obs[6])
        leg1_touch = bool(obs[7])
        all_y_vels.append(y_vel)

        # Check if the lunar lander is impacting for the first time.
        if impact_x_pos is None and (leg0_touch or leg1_touch):
            impact_x_pos = x_pos
            impact_y_vel = y_vel

    # If the lunar lander did not land, set the x-pos to the one from the final
    # timestep, and set the y-vel to the max y-vel (we use min since the lander
    # goes down).
    if impact_x_pos is None:
        impact_x_pos = x_pos
        impact_y_vel = min(all_y_vels)

    if video_env is None:
        env.close()

    return total_reward, impact_x_pos, impact_y_vel


def simulate_HC(sol, seed=None, video_env=None):
    """Simulates the QD Half Cheetah environment.

    Args:
        model (np.ndarray): The array of weights for the linear policy.
        seed (int): The seed for the environment.
        video_env (gym.Env): If passed in, this will be used instead of creating
            a new env. This is used primarily for recording video during
            evaluation.
    Returns:
        total_reward (float): The reward accrued by the lander throughout its
            trajectory.
        rear_foot_contact_time (float): The average time the rear foot is in contact with the
ground. 
        front_foot_contact_time (float): The average time the front foot is in contact with the
ground. 
    """
    
    if video_env is None:
        base_env = gym.make("HalfCheetah-v4", max_episode_steps=300)
        env = QDHalfCheetahWrapper(base_env)
    else:
        env = video_env

    action_dim = env.action_space.shape
    obs_dim = env.observation_space.shape
    
    gin.parse_config_file('config/nnparams.gin')

    model =  MLP(obs_dim, action_dim).deserialize(sol)

    total_reward = 0.0
    rear_foot_contact_time = None
    front_foot_contact_time = None
    obs, _ = env.reset(seed=seed)
    done = False

    while not done:
        action = model.choose_action_cont(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    rear_foot_contact_time = env.desc[0]
    front_foot_contact_time = env.desc[1]

    if video_env is None:
        env.close()

    return total_reward, rear_foot_contact_time, front_foot_contact_time