from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


def get_dataloaders(dataset, batch_size = 8, validation_split = .2, shuffle_dataset = True, random_seed= 42):
    """
    Returns the dataloader objects train_loader and validation_loader. If choosen, shuffels the dataset. Splits the dataset.
    """

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size)) #floor rounds down
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, 
                              batch_size=batch_size, 
                              sampler=train_sampler)
    validation_loader = DataLoader(dataset, 
                                   batch_size=batch_size,
                                   sampler=valid_sampler)
    
    return train_loader, validation_loader