# Importing Libraries
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.datasets import CelebA

# Class Definition
class CelebATarget(Dataset):
    
    def __init__(self, folder_path, transform = transforms.PILToTensor(), target_transform = None, split = "train"):
        self.data = CelebA(root = folder_path, target_type = "attr", transform = transform, target_transform = target_transform, split = split)
        self.hair_pos = [8,9,11,17]
        self.gender_pos = 20
        
    def __getitem__(self, index):
        img, label = self.data[index].to(torch.float)
        y = label[self.hair_pos]
        target = label[self.gender_pos]
        return img, y, target
    
    def __len__(self):
        return len(self.data)