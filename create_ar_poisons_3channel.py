import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import PIL
import argparse
import numpy as np
from autoregressive import ARProcessPerturb3Channel
from autoregressive_params import RANDOM_3C_AR_PARAMS_RNMR_10, RANDOM_100CLASS_3C_AR_PARAMS_RNMR_3

def perturb_with_ar_process(ar_processes, inputs, targets, p_norm, size, crop, eps=1.0):
    """
    input: a (B, 3, 32, 32) tensor where B is the batch size
    output: a (B, 3, 32, 32) tensor with the corresponding AR noise applied and clamped to [0,1] range
    """
    batch_size = inputs.size(0)
    adv_inputs = []
    for i in range(batch_size):
        # choose AR process corresponding to class
        ar_process_perturb = ar_processes[targets[i]]
        
        # create delta of size (3,32,32), scaled to 8/255
        delta, _ = ar_process_perturb.generate(p=p_norm, eps=eps, size=size, crop=crop) # start_signal=inputs[i]
        
        # add delta to image and clamp to [0,1]
        adv_input = (inputs[i] + delta).clamp(0,1)
        adv_inputs.append(adv_input)
    adv_inputs = torch.stack(adv_inputs)
    return adv_inputs

def create_ar_processes(dataset):
    """
    returns a list of 10 AR processes
    """
    if dataset in ['CIFAR10', 'STL10', 'SVHN']:
        b_list = RANDOM_3C_AR_PARAMS_RNMR_10
        print(f'Using {len(b_list)} AR processes for {dataset}')
    elif dataset in ['CIFAR100', 'IMAGENET100']:
        b_list = RANDOM_100CLASS_3C_AR_PARAMS_RNMR_3
        print(f'Using {len(b_list)} AR processes for {dataset}')
    else:
        raise ValueError(f'Dataset {dataset} not supported.')

    ar_processes = []
    for b in b_list:
        ar_p = ARProcessPerturb3Channel(b)
        ar_processes.append(ar_p)

    return ar_processes

class CIFAR100_w_indices(datasets.CIFAR100):
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = PIL.Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

class CIFAR10_w_indices(datasets.CIFAR10):
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = PIL.Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

class STL10_w_indices(datasets.STL10):
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = PIL.Image.fromarray(np.transpose(img, (1,2,0)))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

class SVHN_w_indices(datasets.SVHN):
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = PIL.Image.fromarray(np.transpose(img, (1,2,0)))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index
    
class ImageNetMini_w_indices(datasets.ImageNet):
    def __init__(self, root, split='train', **kwargs):
        super(ImageNetMini_w_indices, self).__init__(root, split=split, **kwargs)
        self.new_targets = []
        self.new_images = []
        for i, (file, cls_id) in enumerate(self.imgs):
            if cls_id <= 99:
                self.new_targets.append(cls_id)
                self.new_images.append((file, cls_id))
        self.imgs = self.new_images
        self.targets = self.new_targets
        self.samples = self.imgs
        print('[class ImageNetMini] num samples:', len(self.samples))
        print('[class ImageNetMini] num targets:', len(self.targets))
        return
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return sample, target, index

def create_poison(args):
    ar_processes = create_ar_processes(args.dataset)

    # Data loading code
    if args.dataset == 'CIFAR10':
        train_dataset = CIFAR10_w_indices(root=os.environ.get('CIFAR_PATH', '/fs/vulcan-datasets/CIFAR'), train=True, download=False, transform=transforms.ToTensor())
        noise_size, noise_crop = (36, 36), 4
    elif args.dataset == 'CIFAR100':
        train_dataset = CIFAR100_w_indices(root=os.environ.get('CIFAR_PATH', '/fs/vulcan-datasets/CIFAR'), train=True, download=False, transform=transforms.ToTensor())
        noise_size, noise_crop = (36, 36), 4
    elif args.dataset == 'STL10':
        train_dataset = STL10_w_indices(root=os.environ.get('STL_PATH', '/vulcanscratch/psando/STL/'), split='train', download=False, transform=transforms.ToTensor())
        noise_size, noise_crop = (100, 100), 4
    elif args.dataset == 'SVHN':
        train_dataset = SVHN_w_indices(root=os.environ.get('SVHN_PATH', '/vulcanscratch/psando/SVHN/'), split='train', download=False, transform=transforms.ToTensor())
        noise_size, noise_crop = (36, 36), 4
    elif args.dataset == 'IMAGENET100':
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()]
        )
        train_dataset = ImageNetMini_w_indices(root=os.environ.get('IMAGENET_PATH', '/vulcanscratch/psando/imagenet/'), split='train', transform=test_transform)
        noise_size, noise_crop = (226, 226), 2
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=False, num_workers=args.workers)


    for batch_idx, batch in enumerate(train_loader):
        inputs, target, indices = batch
        adv_inputs = perturb_with_ar_process(ar_processes, inputs, target, args.p_norm, noise_size, noise_crop, eps=args.epsilon)
        print(f'Exporting at [{batch_idx}/{len(train_loader)}]')
        # Save poison
        export_poison(args, adv_inputs, indices, path=args.save_path)
        
    print('Dataset fully exported.')

def export_poison(args, adv_inputs, indices, path=None):
    if path is None:
        directory = '/fs/nexus-scratch/psando/psando_poisons/paper'
        path = os.path.join(directory, args.noise_name, 'data')
        os.makedirs(path, exist_ok=True)

    def _torch_to_PIL(image_tensor):
        """Torch->PIL pipeline as in torchvision.utils.save_image."""
        image_denormalized = image_tensor
        image_torch_uint8 = image_denormalized.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8)
        image_PIL = PIL.Image.fromarray(image_torch_uint8.numpy())
        return image_PIL

    def _save_image(im, idx, location):
        """Save input image to given location, add poison_delta if necessary."""
        filename = os.path.join(location, str(idx) + '.png')
        _torch_to_PIL(im).save(filename)

    # if mode == 'poison_dataset':
    for adv_input, idx in zip(adv_inputs, indices):
        _save_image(adv_input, idx.item(), location=path)

def main():
    parser = argparse.ArgumentParser(description='Create synthetic poison dataset')
    parser.add_argument('noise_name', type=str, default='', help='Choose the name of the poison')
    parser.add_argument('dataset', type=str, help='Dataset to use: CIFAR10, CIFAR100, STL10, SVHN, or IMAGENET100')
    parser.add_argument('--workers', type=int, default=1, help='Number of data loading workers')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the dataset')
    parser.add_argument('--epsilon', type=float, default=8.0, help='Epsilon for the AR perturbation L-inf (8/255 by default)')
    parser.add_argument('--p_norm', type=int, default=0, help='0 for np.inf, or > 1 for Lp norm')
    args = parser.parse_args()

    if args.p_norm == 0:
        args.epsilon = args.epsilon / 255.0
        args.p_norm = np.inf

    print('Args:\n', args)
    create_poison(args)
        
if __name__ == '__main__':
    main()
