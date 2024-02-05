import numpy as np
import gymnasium as gym
from gymnasium import Wrapper

from gymnasium import spaces
from minigrid.wrappers import ObservationWrapper
from functools import reduce
import operator


class QDHalfCheetahWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.T = 0
        self.tot_reward = 0.0
        self.desc = np.array([0.0 for _ in range(2)])
        self.desc_acc = np.array([0.0 for _ in range(2)])

    def reset(self,seed=None,options=None):
        if seed != None:
            r = super().reset(seed=seed,options=options)
        else:
            r = super().reset()
        self.T = 0
        self.tot_reward = 0.0
        self.desc = np.array([0.0 for _ in range(2)])
        self.desc_acc = np.array([0.0 for _ in range(2)])
        return r

    def step(self, action):
        state, reward, done, trunc, info = super().step(action)
        self.tot_reward += reward
        self.T += 1
        contacts = self.data.contact
        ncon = self.data.ncon
        # The geom ids are found empirically
        floor_id = 0
        rfoot_id = 5
        ffoot_id = 8

        for c in contacts[:ncon]:
            if c.geom1 == floor_id:
                if c.geom2 == rfoot_id:
                    self.desc_acc[0] += 1
                elif c.geom2 == ffoot_id:
                    self.desc_acc[1] += 1
        
        self.desc = self.desc_acc / self.T
        info["bc"] = self.desc
        info["x_pos"] = None
        return state, reward, done, trunc, info


class CustomObsWrapperMG(ObservationWrapper):
    """
    A custom observation wrapper for MiniGrid 
    that converts the image to flat obj_IDs.
    """

    def __init__(self, env):
        """ Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.imgSpace = env.observation_space.spaces["image"]
        imgSize = reduce(operator.mul, self.imgSpace.shape[:-1], 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize,),
            dtype="uint8",
        )

    def observation(self, obs):
        img = obs["image"]
        out = np.zeros(self.imgSpace.shape[:-1], dtype="uint8")
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                out[i, j] = img[i, j, 0]
        return out.flatten()