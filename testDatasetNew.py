import os
import glob
import numpy as np
import re
import scipy.io.wavfile
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder


class TestDatasetNew(data.Dataset):
    """Pytorch dataset for instruments
    args:
        root: root dir containing an audio directory with wav files.
        transform (callable, optional): A function/transform that takes in
                a sample and returns a transformed version.
    """

    def __init__(self, root, root_fft, transform=None):
        assert(isinstance(root, str))

        self.root = root
        self.root_fft = root_fft
        self.filenames = glob.glob(os.path.join(root, "audio/*.wav"))           
        self.transform = transform
            
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        name = self.filenames[index]
        _, sample = scipy.io.wavfile.read(name)
        
        fft = np.load( root_fft + 'fft_' + str(index) + '.npy')
        
        if self.transform is not None:
            sample = self.transform(sample)
        return [sample, fft]
