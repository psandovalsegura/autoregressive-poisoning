import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision import datasets, transforms
from PIL import Image

def show(imgs, figsize=(15,15), save_filename=None):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=figsize)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if save_filename is not None:
        print('Saved to {}'.format(save_filename))
        fig.savefig(save_filename)

def batch_min(x):
    m = [torch.min(xp) for xp in x]
    return torch.tensor(m)

def batch_max(x):
    m = [torch.max(xp) for xp in x]
    return torch.tensor(m)

def batch_normalize(x):
    _max = batch_max(x)[:,None,None,None]
    _min = batch_min(x)[:,None,None,None]
    return (x - _min) / (_max - _min)

class AdversarialPoisonWithClean(torch.utils.data.Dataset):
    def __init__(self, root, dataset_name):
        self.baseset = get_baseset(dataset_name=dataset_name)
        self.transform = self.baseset.transform
        self.samples = os.listdir(os.path.join(root, 'data'))
        self.root = root

    def __len__(self):
        return len(self.baseset)

    def __getitem__(self, idx):
        true_index = int(self.samples[idx].split('.')[0])
        true_img, label = self.baseset[true_index]
        return self.transform(Image.open(os.path.join(self.root, 'data',
                                            self.samples[idx]))), label, true_img

def get_baseset(dataset_name):
    if dataset_name == 'STL10':
        trainset = datasets.STL10(root='/vulcanscratch/psando/STL', split='train', download=False, transform=transforms.ToTensor())
    elif dataset_name == 'CIFAR10':
        trainset = datasets.CIFAR10(root='/vulcanscratch/psando/cifar-10/', train=True, download=False, transform=transforms.ToTensor())
    else:
        raise ValueError('Dataset {} not supported'.format(dataset_name))
    return trainset