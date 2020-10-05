"""
Created on Thu Aug 20 15:18:04 2020

@author: biopython
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class PlantDataset(Dataset):
    
    
    """Loads the Image data and zips it with corresponding probabilities
    as labels.
    
    Args:
        transforms_list - list of transformer objects produced from
    torchvision.transforms.
        path - main path to folder containing the images folder and datafile
    
    Returns:
        Dataset Object
    """
    
    
    def __init__(self, transforms_list=None, path=None, filename=None):
        
        self.transforms = transforms.Compose(transforms_list)
        self.df_path = path
        df = pd.read_csv(path+'/'+filename)
        leaf_probs = np.array(df.drop('image_id', axis=1))
        self.df = pd.concat(
            [df['image_id'],
             pd.Series(np.argwhere(leaf_probs == 1)[:, 1], name='label')],axis=1)



    def __len__(self):   # gets num of Images
        return self.df.shape[0]

    def __getitem__(self, idx):   # selects image by index
        image_path = (self.df_path
                      + '/images/'
                      + self.df.loc[idx, 'image_id']
                      + '.jpg')

        image = Image.open(image_path)

        if self.transforms:
            image = self.transforms(image)
        
        label = np.array(self.df.iloc[idx, 1])
        label = torch.from_numpy(label)
        
        return image, label

def viewImages(dataloader, num_of_images,
               std=[1.0, 1.0, 1.0], mean=[0.0, 0.0, 0.0]):
    images, labels = next(iter(dataloader))
    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    # display 20 training images
    for idx in np.arange(num_of_images):
        ax = fig.add_subplot(2, num_of_images//2, idx+1, xticks=[], yticks=[])
        img = np.moveaxis(images[idx].numpy(), 0, 2)
        plt.imshow(
            torch.clamp(
                torch.from_numpy(img) * np.array(std) + np.array(mean), 0.0, 1.0))
        # ax.set_title(condition[labels[idx]])
        fig.suptitle('Sample Training Images', fontsize=14)
