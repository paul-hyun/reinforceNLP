
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
(Value Network
"""
class ValueNet(nn.Module):
    def __init__(self, n_state, n_action):
        super(ValueNet, self).__init__()

        self.layer1 = self.linear(n_state, 24)
        self.output = self.linear(24, n_action)
    
    def linear(self, n_input, n_output):
        layer = nn.Linear(n_input, n_output)
        nn.init.xavier_normal_(layer.weight)
        return layer
    
    def forward(self, state):
        output = F.relu(self.layer1(state))
        output = self.output(output)
        return output


"""
Ploicy Network
"""
class PolicyNet(nn.Module):
    def __init__(self, n_state, n_action):
        super(PolicyNet, self).__init__()
        
        self.layer1 = self.linear(n_state, 24)
        self.output = self.linear(24, n_action)

    def linear(self, n_input, n_output):
        layer = nn.Linear(n_input, n_output)
        nn.init.xavier_uniform_(layer.weight)
        return layer
    
    def forward(self, state):
        output = F.relu(self.layer1(state))
        output = F.softmax(self.output(output), dim=1)
        return output


"""
Configuration
"""
class Config(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

