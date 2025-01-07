# %%
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# %%
data_transforms = {
    "train":transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # -> [0, 1]
        transforms.Normalize(0.5, 0.5, 0.5)
    ]),
    "val":transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(0.5,0.5, 0.5)
    ])
}
# %%
root_dir = 'afhq'
dataset = {
    x : datasets.ImageFolder(os.path.join(root_dir, x), data_transforms[x]) for x in ["train", "val"]
}

dataset_loader = {
    x : DataLoader(dataset[x], batch_size=8, shuffle=True) for x in ["train", "val"] 
}
dataset_sizes = {x : len(dataset[x]) for x in ["train", "val"]}
# %%
