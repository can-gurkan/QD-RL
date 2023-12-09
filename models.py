import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):

    def __init__(self, n_observations, n_actions, fc1_dims, fc2_dims):
        super(MLP, self).__init__()
        self.input_dims = n_observations
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions
    
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
        #state = T.tensor([np.array(observation, dtype=np.float32)]).to(self.device)
        state = T.from_numpy(observation.astype(np.float32)).to(self.device)
        with T.no_grad():
            actions = self.forward(state)
        #print(actions)
        action = T.argmax(actions).item()
        return action