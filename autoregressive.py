import torch
import torch.nn.functional as F
import numpy as np

class ARProcessPerturb3Channel():
    def __init__(self, b=None):
        """
        b: None or an numpy array of shape (3,3) where the last entry is 0.
            The
        """
        super().__init__()
        if b is None:
            # Random AR process parameters are more stable when normalized
            self.b = torch.randn((3,3,3))
            for c in range(3):
                self.b[c][2][2] = 0
                self.b[c] /= torch.sum(self.b[c])
        # check if b is a torch tensor
        elif type(b) == torch.Tensor:
            self.b = b
        else:
            self.b = torch.tensor(b).float()
        self.num_channels = 3

    def generate(self, size=(32,32), eps=(8/255), crop=0, p=np.inf, gaussian_noise=False):
        """
        returns a (num_channels, 32, 32) tensor with p-norm of eps
        updates the start_signal using the ar process parameters

        Params:
            size: size of the generated perturbation, which may be cropped
            crop: number of rows and columns to crop
            p: p-norm of the generated perturbation
            gaussian_noise: whether to add gaussian noise to the ar process (unrelated to start signal)
        """
        start_signal = torch.randn((self.num_channels, size[0], size[1]))
        kernel_size = 3
        rows_to_update = size[0] - kernel_size + 1
        cols_to_update = size[1] - kernel_size + 1

        ar_coeff = self.b.unsqueeze(dim=1) # (3, 1, 3, 3)
        for i in range(rows_to_update):
            for j in range(cols_to_update):
                val = torch.nn.functional.conv2d(start_signal[:,i:i+kernel_size,j:j+kernel_size], ar_coeff, groups=self.num_channels)
                
                # update entry
                noise = torch.randn(1) if gaussian_noise else 0
                start_signal[:, i+kernel_size-1, j+kernel_size-1] = val.squeeze() + noise
        
        start_signal_crop = start_signal[:,crop:, crop:]

        # Scale perturbation to be of size eps in Lp norm
        generated_norm = torch.norm(start_signal_crop, p=p, dim=(0,1,2))
        scale = (1/generated_norm) * eps
        start_signal_crop = scale * start_signal_crop
        
        return start_signal_crop, generated_norm

    def get_filter(self):
        """
        return the (1,3,3,3) filter which responds to this AR process
        """
        # the matching filter is almost identical to the AR
        # process parameters, but has a -1 as the last entry
        filter = self.b.detach().clone()
        for c in range(3):
            filter[c][2][2] = -1

        # because generated filters have 8 entries that sum to 1,
        # the addition of -1 in the final entry, ensures the sum of 
        # filter entries is 0
        return filter.unsqueeze(dim=0)

    def __repr__(self) -> str:
        return f'{self.b.numpy()}'

def response(filter, signal, response_fn=(lambda x: F.relu(x).sum().item())):
    """
    filter: a (1,3,3,3) tensor representing the filter
    signal: a (3,n,n) tensor representing the signal
    """
    signal = signal.unsqueeze(dim=0)
    conv_output = torch.nn.functional.conv2d(signal, filter)
    response = response_fn(conv_output)
    return response