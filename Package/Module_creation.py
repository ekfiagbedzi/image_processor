"""
Created on Tue Aug 25 18:12:53 2020

@author: biopython
"""
# load and split data into training, validation and testing images

# loading and preprocessing train labels

from DatasetCreater import PlantDataset, viewImages
from torch.utils.data import DataLoader
from torchvision import transforms

path = '/home/biopython/Downloads/Datasets/plant-pathology-2020-fgvc7'



transforms=[transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomResizedCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]

# mean and std for normalizing and denormallizing
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_data = PlantDataset(path=path, transforms_list=transforms, filename='test.csv')
train_loader = DataLoader(train_data, batch_size=20)
viewImages(train_loader, 10, std=std, mean=mean)
