
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Cartpole Value (Q) Network
"""
class CartpoleQ(nn.Module):
    def __init__(self, n_state, n_action):
        super(CartpoleQ, self).__init__()

        self.layer1 = self.linear(n_state, 256)
        self.layer2 = self.linear(256, 256)
        self.output = self.linear(256, n_action)
    
    def linear(self, n_input, n_output):
        layer = nn.Linear(n_input, n_output)
        nn.init.xavier_uniform_(layer.weight)
        return layer
    
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
        
        self.layer1 = self.linear(n_state, 256)
        self.layer2 = self.linear(256, 256)
        self.output = self.linear(256, n_action)

    def linear(self, n_input, n_output):
        layer = nn.Linear(n_input, n_output)
        # nn.init.xavier_uniform_(layer.weight)
        return layer
    
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

