# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np

class ChessResNet(nn.Module):
    # ------------------ CHANGE STARTS HERE ------------------
    def __init__(self, game, args):
        super(ChessResNet, self).__init__()
        
        # Extract params from the args dictionary
        self.num_res_blocks = args.get('num_res_blocks', 5) # Default to 5 if missing
        self.num_channels = args.get('num_channels', 256)   # Default to 256 if missing
        
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        
        # Initial convolution block
        self.conv_input = nn.Sequential(
            nn.Conv2d(12, self.num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.num_channels),
            nn.ReLU()
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(self.num_channels) for _ in range(self.num_res_blocks)
        ])
        
        # Policy Head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(num_channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU()
        )
        self.policy_fc = nn.Linear(2 * self.board_x * self.board_y, self.action_size)
        
        # Value Head
        self.value_conv = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.value_fc = nn.Sequential(
            nn.Linear(1 * self.board_x * self.board_y, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh() # Output value between -1 and 1
        )

    def forward(self, s):
        # s: batch_size x 12 x 8 x 8
        x = self.conv_input(s)
        
        for block in self.res_blocks:
            x = block(x)
            
        # Policy Head
        p = self.policy_conv(x)
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1)
        
        # Value Head
        v = self.value_conv(x)
        v = v.view(v.size(0), -1)
        v = self.value_fc(v)
        
        return p, v

class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out