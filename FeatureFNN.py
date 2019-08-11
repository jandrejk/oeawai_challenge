import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from math_utils import logMagStft, ffts
from scipy.signal import hilbert


# NN architecture (three conv and two fully connected layers)
class FeatureFNN(nn.Module):
    def __init__(self, device):
        super(FeatureFNN, self).__init__()
        self.fc1_env = nn.Linear(1280, 1024)
        self.fc2_env = nn.Linear(1024, 512)
        self.fc1_fft = nn.Linear(1280, 1024)
        self.fc2_fft = nn.Linear(1024, 512)
        self.fc1_together = nn.Linear(1024, 512)
        self.fc2_together = nn.Linear(512, 10)        
        self.device = device
        self.pool = nn.MaxPool1d(50)

    def forward(self, x):
        
        # create env
        x_env = np.zeros([len(x), 64000])
        for index, audio in enumerate(x.cpu().numpy()):
            x_env[index] = np.abs(hilbert(audio))
        
        x_env = torch.from_numpy(x_env).view(len(x_env),1,-1).to(self.device).float()
        x_env = self.pool(x_env).view(len(x_env), -1)
        
        # fft
        x_fft = np.zeros([len(x), 64000])
        for index, audio in enumerate(x.cpu().numpy()):
            x_fft[index] = np.abs(ffts(audio))
        
        x_fft = torch.from_numpy(x_fft).view(len(x_env),1,-1).to(self.device).float()
        x_fft = self.pool(x_fft).view(len(x_env), -1)
        
        # x.size is (batch_size, 1280)
        x_env = F.relu(self.fc1_env(x_env))
        x_env = F.relu(self.fc2_env(x_env))

        x_fft = F.relu(self.fc1_env(x_fft))
        x_fft = F.relu(self.fc2_env(x_fft))
        
        x_together = torch.cat([x_env,x_fft], dim = 1)
        
        x_together = F.relu(self.fc1_together(x_together))
        x_together = self.fc2_together(x_together)

        return F.log_softmax(x_together, dim=1)