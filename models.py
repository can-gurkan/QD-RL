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
    
    def choose_action(self, observation):
        state = T.tensor([np.array(observation, dtype=np.float32)]).to(self.device)
        with T.no_grad():
            actions = self.forward(state)
        action = T.argmax(actions).item()
        return action