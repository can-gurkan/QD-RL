import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gin


@gin.configurable
class MLP(nn.Module):

    def __init__(self, n_observations, n_actions, layer_shapes=gin.REQUIRED):
        super(MLP, self).__init__()

        #layer_shapes = (64,64)
        layers = ([np.product(n_observations)] + list(layer_shapes) + [np.product(n_actions)])
        self._layer_shapes = list(zip(layers[:-1], layers[1:]))

        layers = []
        for i, shape in enumerate(self._layer_shapes):
            layers.append(nn.Linear(*shape))
            if i != len(layer_shapes) - 1:
                layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        """Computes actions for a batch of observations."""
        return self.model(x)
    
    def serialize(self):
        """Returns 1D array with all parameters in the nn."""
        return np.concatenate([p.data.cpu().detach().numpy().ravel() for p in self.parameters()])

    def deserialize(self, array):
        """Loads parameters from 1D array."""
        array = np.copy(array)
        arr_idx = 0
        for param in self.parameters():
            shape = tuple(param.data.shape)
            length = np.product(shape)
            block = array[arr_idx:arr_idx + length]
            if len(block) != length:
                raise ValueError("Array not long enough!")
            block = np.reshape(block, shape)
            arr_idx += length
            param.data = T.from_numpy(block).float()
        return self

    def gradient(self):
        """Returns 1D array with gradient of all parameters in the actor."""
        return np.concatenate([p.grad.cpu().detach().numpy().ravel() for p in self.parameters()])

    def choose_action(self, observation):
        state = T.from_numpy(observation.astype(np.float32)).to(self.device)
        with T.no_grad():
            actions = self.forward(state)
        action = T.argmax(actions).item()
        return action