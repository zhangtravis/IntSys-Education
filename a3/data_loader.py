import os
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch
from torch import Tensor

class SimpleDataset(Dataset):
    """SimpleDataset [summary]
    
    [extended_summary]
    
    :param path_to_pkl: Path to PKL file with Images
    :type path_to_pkl: str
    :param path_to_labels: path to file with labels
    :type path_to_labels: str
    """
    def __init__(self, path_to_pkl, path_to_labels):
        ## TODO: Add code to read csv and load data. 
        ## You should store the data in a field.
        # Eg (on how to read .csv files):
        # with open('path/to/.csv', 'r') as f:
        #   lines = ...
        ## Look up how to read .csv files using Python. This is common for datasets in projects.

        # Load images
        with open(path_to_pkl, 'rb') as file:
            self.images = pickle.load(file)
        # Load labels
        with open(path_to_labels, 'rb') as file:
            self.labels = pickle.load(file)

    def __len__(self):
        """__len__ [summary]
        
        [extended_summary]
        """
        ## TODO: Returns the length of the dataset.
        return len(self.images)

    def __getitem__(self, index):
        """__getitem__ [summary]
        
        [extended_summary]
        
        :param index: [description]
        :type index: [type]
        """
        ## TODO: This returns only ONE sample from the dataset, for a given index.
        ## The returned sample should be a tuple (x, y) where x is your input 
        ## vector and y is your label
        ## Before returning your sample, you should check if there is a transform
        ## sepcified, and pply that transform to your sample
        # Eg:
        # if self.transform:
        #   sample = self.transform(sample)
        ## Remember to convert the x and y into torch tensors.

        x = np.array(self.images[index])
        
        y = np.array(self.labels[index])
        y = torch.tensor(y, dtype=torch.float)
        x = Tensor(x).view(1, 28, 28).float()
        return x , y


def get_data_loaders(path_to_pkl, 
                     path_to_labels,
                     train_val_test=[0.8, 0.2, 0.2], 
                     batch_size=32):
    """get_data_loaders [summary]
    
    [extended_summary]
    
    :param path_to_csv: [description]
    :type path_to_csv: [type]
    :param train_val_test: [description], defaults to [0.8, 0.2, 0.2]
    :type train_val_test: list, optional
    :param batch_size: [description], defaults to 32
    :type batch_size: int, optional
    :return: [description]
    :rtype: [type]
    """
    # First we create the dataset given the path to the .csv file
    dataset = SimpleDataset(path_to_pkl, path_to_labels)

    # Then, we create a list of indices for all samples in the dataset.
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    ## TODO: Rewrite this section so that the indices for each dataset split
    ## are formed. You can take your code from last time

    ## BEGIN: YOUR CODE
    train_split, validation_split, test_split = train_val_test[0], train_val_test[1], train_val_test[2]
    # Generate split between train and test data
    train_test_split = int(np.floor(train_split * dataset_size))
    # Shuffle indices
    np.random.shuffle(indices)
    # Generate split within train data for train and val data (train_val_split has 80% of train_val_data for training, 20% of train_val_data for val data
    train_val_split = int(np.floor(train_split * train_test_split))
    # Generate list of indices for train and test data
    train_val_indices = indices[:train_test_split]
    train_indices = train_val_indices[:train_val_split]
    val_indices = train_val_indices[train_val_split:]
    test_indices = indices[train_test_split:]
    ## END: YOUR CODE

    # Now, we define samplers for each of the train, val and test data
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # Testing purposes
    train_loader, val_loader, test_data = get_data_loaders('data/processedImages.pkl', 'data/processedLabels.pkl')

    for batch_index, (x, y) in enumerate(train_loader):
        print(f"Batch {batch_index}")
        print(f"X: {x.shape}")
        print(f"Y: {y.shape}")

        if batch_index == 1000:
            break

