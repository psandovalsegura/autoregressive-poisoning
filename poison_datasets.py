import os
import torch
import numpy as np
from PIL import Image

class AdversarialPoison(torch.utils.data.Dataset):
    def __init__(self, root, baseset):
        self.baseset = baseset
        self.transform = self.baseset.transform
        self.samples = os.listdir(os.path.join(root, 'data'))
        self.root = root

        # Load images into memory to prevent IO from disk
        self.data, self.targets = self._load_images()

    def __len__(self):
        return len(self.baseset)

    def __getitem__(self, idx):
        return self.transform(self.data[idx]), self.targets[idx]

    def _load_images(self):
        data = []
        targets = []
        num_data_to_load = len(self.baseset)
        for i in range(num_data_to_load):
            true_index = int(self.samples[i].split('.')[0])
            _, label = self.baseset[true_index]
            data.append(Image.open(os.path.join(self.root, 'data', self.samples[i])).copy())
            targets.append(label)
        return data, targets

class UnlearnablePoison(torch.utils.data.Dataset):
    def __init__(self, root, baseset):
        self.baseset = baseset
        self.root = root
        self.classwise = 'classwise' in root
        noise = torch.load(root)

        # Load images into memory to prevent IO from disk
        self._perturb_images(noise)

    def __len__(self):
        return len(self.baseset)

    def __getitem__(self, idx):
        return self.baseset[idx]

    def _perturb_images(self, noise):
        if 'stl' in self.root.lower() or 'svhn' in self.root.lower():
            perturb_noise = noise.mul(255).clamp_(-255, 255).to('cpu').numpy()
        else:
            perturb_noise = noise.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        self.baseset.data = self.baseset.data.astype(np.float32)
        for i in range(len(self.baseset)):
            if self.classwise:
                self.baseset.data[i] += perturb_noise[self.baseset.targets[i]]
            else: # samplewise
                self.baseset.data[i] += perturb_noise[i]
            self.baseset.data[i] = np.clip(self.baseset.data[i], a_min=0, a_max=255)
        self.baseset.data = self.baseset.data.astype(np.uint8)
        