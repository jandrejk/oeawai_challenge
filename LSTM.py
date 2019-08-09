import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from math_utils import logMagStft


class LSTM(nn.Module):
    def __init__(self, device, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)
        self.device = device
    
    def forward(self, x):
        n_fft = 510
    
        spectrograms = np.zeros((len(x), n_fft//2+1, int(2*64000/n_fft)+2))
        for index, audio in enumerate(x.cpu().numpy()):
            spectrograms[index] = logMagStft(audio, 16000, n_fft)
        
        x = torch.from_numpy(spectrograms).to(self.device).float()        
        
        #print(x.size())
        
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        #out, _ = self.lstm(x) # alternatively, without setting h0 and c0
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return F.log_softmax(out, dim=1)