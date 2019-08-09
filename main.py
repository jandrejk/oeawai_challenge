#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[1]:

print('IMPORTS')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
import IPython.display
from IPython.display import Audio
import torch.optim as optim
from types import SimpleNamespace
import scipy.signal as sc
import time
from sklearn.metrics import f1_score

from trainDataset import TrainDataset
from testDataset import TestDataset
#from trainDatasetNew import TrainDatasetNew
#from testDatasetNew import TestDatasetNew
from validation_split import get_dataloaders
from math_utils import logMagStft, ffts
from SpectrogramCNN import SpectrogramCNN
from train_utils import train, test
from evaluation_utils import get_mean_F1
from MulitScale1DResNet import MSResNet
from SpectralResNet import SpectralResNet34
from LSTM import LSTM


# ### Parameters

# In[2]:

print('SETTING PARAMETERS')

validation_split = .2
do_plots = False
load_model = False
args = SimpleNamespace(batch_size=8, test_batch_size=32, epochs=5,
                       lr=0.01, momentum=0.5, seed=1, log_interval=200, 
                      net = LSTM) #SpectrogramCNN, MSResNet, SpectralResNet, SpectralMSResNet, LSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available(): # adapt those paths on other machine
    print('no cuda')
    path_train = './../data/train-small/'
    path_test =  './../data/test/kaggle-test/'
else:
    print('with cuda')
    path_train = './../data/kaggle-train/'
    path_test =  './../data/kaggle-test/'
    
path_model = 'models/model.pt'
path_submission = 'submissions/'
    
sample_rate = 16000
nmbr_classes = 10


# ### Original Dataset

# In[3]:

print('LOADING DATASETS')

toFloat = transforms.Lambda(lambda x: x / np.iinfo(np.int16).max)

trainDataset = TrainDataset(path_train, transform=toFloat)
print(len(trainDataset))

testDataset = TestDataset(path_test, transform=toFloat)
print(len(testDataset))


# In[4]:


input_size = len(trainDataset[0][0])
print('input size: ',input_size)


# ### Dataloaders

# In[12]:


# validation split is done here

print('CREATING DATALOADERS')

train_loader, validation_loader = get_dataloaders(trainDataset, 
                                                  batch_size = args.batch_size, 
                                                  validation_split = validation_split, 
                                                  shuffle_dataset = True, 
                                                  random_seed = None)

for samples, instrument_family_target in train_loader:
        print(samples.shape, instrument_family_target.shape,
              instrument_family_target.data)
        print(torch.min(samples), torch.max(samples))
        print(trainDataset.transformInstrumentsFamilyToString(instrument_family_target.data))
        break
        
for samples, instrument_family_target in validation_loader:
        print(samples.shape, instrument_family_target.shape,
              instrument_family_target.data)
        print(torch.min(samples), torch.max(samples))
        print(trainDataset.transformInstrumentsFamilyToString(instrument_family_target.data))
        break


# In[13]:


test_loader = data.DataLoader(testDataset, batch_size=args.batch_size, shuffle=False) #!!! shuffle should be false
for samples in test_loader:
        print(samples[0].shape)
        print(torch.min(samples[0]), torch.max(samples[0]))
        break


# ### Main

# In[14]:

print('CREATING MODEL')

if (args.net == SpectrogramCNN) or (args.net == SpectralResNet34):
    model = args.net(device).to(device)


# In[15]:


if (args.net == MSResNet):# or (args.net == SpectralMSResNet):
    model = args.net(1, device).to(device)


# In[16]:


if args.net == LSTM:
    model = args.net(device, input_size = 252, hidden_size = 320, num_layers = 1, num_classes = 10).to(device)


# In[17]:


print(model)


# In[18]:


# Main
optimizer = optim.Adam(model.parameters(), lr=args.lr)

info = {'highest F1' : 100,
        'saved epoch' : None}


# In[19]:

print('STARTING TRAINING')

for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch, start_time = time.time())
    f1 = get_mean_F1(model, validation_loader)
    print('after epoch {} got f1 score of {}'.format(epoch , f1))
    if f1 > info['highest F1']:
        info['highest F1'] = np.copy(f1)
        info['saved epoch'] = epoch 
        test(args, model, device, test_loader, epoch, trainDataset, testDataset, path_submission)
        torch.save(model, path_model)
        print('currently best model --> saved')
 
print('TRAINING DONE')
print(info)


# ### Load Model

# In[ ]:


#if load_model:
#    model = torch.load(path_model)


# In[ ]:


#get_mean_F1(model, validation_loader)


# In[ ]:


#epoch=10
#test(args, model, device, test_loader, epoch, trainDataset, testDataset, path_submission)


# In[ ]:




