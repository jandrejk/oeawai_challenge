import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from math_utils import logMagStft


# NN architecture (three conv and two fully connected layers)
class SpectrogramCNN(nn.Module):
    def __init__(self, device):
        super(SpectrogramCNN, self).__init__()
        self.first_conv = nn.Conv2d(1, 20, 5, 1)
        self.second_conv = nn.Conv2d(20, 50, 5, 2)
        self.third_conv = nn.Conv2d(50, 50, 5, 2)        
        self.fc1 = nn.Linear(50*6*6, 500)
        self.fc2 = nn.Linear(500, 10)
        self.device = device

    def forward(self, x):
        n_fft = 510
    
        spectrograms = np.zeros((len(x), n_fft//2+1, int(2*64000/n_fft)+2))
        for index, audio in enumerate(x.cpu().numpy()):
            spectrograms[index] = logMagStft(audio, 16000, n_fft)
        
        x = torch.from_numpy(spectrograms[:, np.newaxis, :, :]).to(self.device).float()
        
        # x.size is (batch_size, 1, 256, 252)
        x = F.relu(self.first_conv(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.second_conv(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.third_conv(x))
        x = F.max_pool2d(x, 2, 2)
        # x.size is (batch_size, 50, 6, 6)
        x = x.view(-1, 6*6*50)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)