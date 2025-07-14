import torch

def basic_augmentations(image, label):
    if torch.rand(1) < 0.5:
        image = torch.flip(image, dims=[2])
        label = torch.flip(label, dims=[1])
    if torch.rand(1) < 0.5:
        image = torch.flip(image, dims=[1])
        label = torch.flip(label, dims=[0])
    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
        image = torch.rot90(image, k=k, dims=[1,2])
        label = torch.rot90(label, k=k, dims=[0,1])
    return image, label
