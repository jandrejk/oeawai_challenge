import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from math_utils import logMagStft


# NN architecture 
class CNN1D(nn.Module):
    def __init__(self, device):
        super(CNN1D, self).__init__()
        
        self.conv_1 = nn.Sequential(
          nn.Conv1d(1, 16, 9, 1),
          nn.ReLU(),
          nn.Conv1d(16, 16, 9, 1),
          nn.ReLU(),
          nn.MaxPool1d(16),
          nn.Dropout(p=0.1)
        )
        
        self.conv_2 = nn.Sequential(
          nn.Conv1d(16, 32, 3, 1),
          nn.ReLU(),
          nn.Conv1d(32, 32, 3, 1),
          nn.ReLU(),
          nn.MaxPool1d(4),
          nn.Dropout(p=0.1)
        )
        
        self.conv_3 = nn.Sequential(
          nn.Conv1d(32, 32, 3, 1),
          nn.ReLU(),
          nn.Conv1d(32, 32, 3, 1),
          nn.ReLU(),
          nn.MaxPool1d(4),
          nn.Dropout(p=0.1)
        )
        
        self.conv_4 = nn.Sequential(
          nn.Conv1d(32, 256, 3, 1),
          nn.ReLU(),
          nn.Conv1d(256, 256, 3, 1),
          nn.ReLU(),
          nn.MaxPool1d(244),
          nn.Dropout(p=0.2)
        )
        
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64,1028)
        self.out = nn.Linear(1028,10)
        
        self.device = device
        

    def forward(self, x):
        x = x.to(self.device).float().view(len(x),1,-1)
        
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        
        x = x.view(len(x), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        
        return F.log_softmax(x, dim=1)
    
    
# NN architecture 
class CNN2D(nn.Module):
    def __init__(self, device):
        super(CNN2D, self).__init__()
        
        self.conv_1 = nn.Sequential(
          nn.Conv2d(1, 32, (4,10), 1),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.MaxPool2d((2,2))
        )
        
        self.conv_2 = nn.Sequential(
          nn.Conv2d(32, 32, (4,10), 1),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.MaxPool2d((2,2))
        )
        
        self.conv_3 = nn.Sequential(
          nn.Conv2d(32, 32, (4,10), 1),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.MaxPool2d((2,2))
        )
        
        self.conv_4 = nn.Sequential(
          nn.Conv2d(32, 32, (4,10), 1),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.MaxPool2d((2,2))
        )

        self.fc1 = nn.Linear(32*13*7, 64)
        self.batchnorm = nn.BatchNorm1d(1)
        self.out = nn.Linear(64,10)
        
        self.device = device
        

    def forward(self, x):
        n_fft = 510
    
        spectrograms = np.zeros((len(x), n_fft//2+1, int(2*64000/n_fft)+2))
        for index, audio in enumerate(x.cpu().numpy()):
            spectrograms[index] = logMagStft(audio, 16000, n_fft)
        
        x = torch.from_numpy(spectrograms[:, np.newaxis, :, :]).to(self.device).float()
        
        
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        
        x = x.view(len(x), -1)
        
        x = self.fc1(x)
        x = self.batchnorm(x.view(len(x),1,-1)).view(len(x),-1)
        x = F.relu(x)
        x = self.out(x)
        
        return F.log_softmax(x, dim=1)