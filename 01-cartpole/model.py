
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Cartpole Network
"""
class CartpoleNet(nn.Module):
    def __init__(self, n_state, n_action, softmax=True):
        super(CartpoleNet, self).__init__()
        
        self.softmax = softmax
        self.layer1 = nn.Linear(n_state, 256)
        self.layer2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, n_action)
    
    def forward(self, state):
        output = F.relu(self.layer1(state))
        output = F.relu(self.layer2(output))
        output = self.output(output)
        return F.softmax(output, dim=1) if self.softmax else output


"""
Cartpole Configuration
"""
class CartpoleConfig(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

