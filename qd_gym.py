import numpy as np
import gymnasium as gym
from gymnasium import Wrapper


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
    