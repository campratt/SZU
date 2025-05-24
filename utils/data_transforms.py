import torch
from torchvision import transforms
import numpy as np
import os

module_dir = os.path.dirname(__file__)
freq_norms = np.genfromtxt(os.path.join(module_dir,'freq_norms_PR4.txt')).T
sz_norms = np.genfromtxt(os.path.join(module_dir,'sz_norms.txt'))



class TransformSZ(object):
    def __init__(self, N_dim, eval=False):

        flip = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            
        ])
        normalize_x = transforms.Compose([
            transforms.Normalize(tuple(freq_norms[0]), tuple(freq_norms[1])),
        ])


        normalize_y = transforms.Compose([
            transforms.Normalize((sz_norms[0]), (sz_norms[1])),
        ])

        
        crop_center = transforms.Compose([
            transforms.CenterCrop(N_dim),
        ])

        crop_rand = transforms.Compose([
            transforms.RandomCrop(N_dim),
        ])

        if eval:
            self.transformation_x = transforms.Compose([
            normalize_x,
            crop_center,
            ])

            self.transformation_y = transforms.Compose([
            normalize_y,
            crop_center,
            ])

            self.transformation_y_mask = transforms.Compose([
            crop_center,
            ])


        else:
            self.transformation_x = transforms.Compose([
            flip,
            normalize_x,
            crop_rand,
            ])

            self.transformation_y = transforms.Compose([
            flip,
            normalize_y,
            crop_rand,
            ])

            self.transformation_y_mask = transforms.Compose([
            flip,
            crop_rand,
            ])


        


    def __call__(self, x_image, y_image=None, mask_image=None):
        state = torch.get_rng_state()
        x = self.transformation_x(x_image)
        torch.set_rng_state(state)

        if y_image is None:
            return x, None, None
        elif mask_image is not None:
            y = self.transformation_y(y_image)
            mask = self.transformation_y_mask(mask_image)
            return x, y, mask
        else:
            y = self.transformation_y(y_image)
            return x, y, None
