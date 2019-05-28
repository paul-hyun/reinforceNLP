
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Cartpole Value (Q) Network
"""
class CartpoleQ(nn.Module):
    def __init__(self, n_state, n_action):
        super(CartpoleQ, self).__init__()

        self.layer1 = nn.Linear(n_state, 256)
        self.layer2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, n_action)
    
    def forward(self, state):
        output = F.relu(self.layer1(state))
        output = F.relu(self.layer2(output))
        output = self.output(output)
        return output


"""
Cartpole Policy Network
"""
class CartpoleP(nn.Module):
    def __init__(self, n_state, n_action):
        super(CartpoleP, self).__init__()
        
        self.layer1 = nn.Linear(n_state, 256)
        self.layer2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, n_action)
    
    def forward(self, state):
        output = F.relu(self.layer1(state))
        output = F.relu(self.layer2(output))
        output = F.softmax(self.output(output), dim=1)
        return output


"""
Cartpole Configuration
"""
class CartpoleConfig(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

