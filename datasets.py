import torch
import numpy as np

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, IDs, x_dir, y_dir, mask_dir=None, N_per_epoch = None, transform=None):
        super(TrainDataset).__init__()
        
        self.IDs = IDs
        self.transform = transform
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.mask_dir = mask_dir

        if N_per_epoch is not None:
            self.Len = N_per_epoch
        else:
            self.Len = len(self.IDs)


    def __len__(self):
        return self.Len
    
    

    def __getitem__(self, idx):
        # Get sample
        ID = self.IDs[idx]
        
        path_x = self.x_dir + f'x_{ID}.npy'
        x = torch.from_numpy(np.load(path_x)).float()

        path_y = self.y_dir + f'y_{ID}.npy'
        y = torch.from_numpy(np.load(path_y)).float().unsqueeze(0)

        if self.mask_dir is not None:
            path_mask = self.mask_dir + f'mask_{ID}.npy'
            mask = torch.from_numpy(np.load(path_mask)).float().unsqueeze(0)
        else:
            mask = None
        
        
        if self.transform is not None:
            x, y, mask = self.transform(x, y, mask)

        return [x, y, mask]
    

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, IDs, x_dir, y_dir=None, mask_dir=None, transform=None):
        super(EvalDataset).__init__()
        self.IDs = IDs
        self.transform = transform
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.mask_dir = mask_dir

        
        self.Len = len(self.IDs)


    def __len__(self):
        return self.Len
    
    

    def __getitem__(self, idx):
        # Get random sample

        
        ID = self.IDs[idx]
        
        path_x = self.x_dir + f'x_{ID}.npy'
        x = torch.from_numpy(np.load(path_x)).float()

        if self.y_dir is not None and self.mask_dir is not None:
            path_y = self.y_dir + f'y_{ID}.npy'
            y = torch.from_numpy(np.load(path_y)).float().unsqueeze(0)

            path_mask = self.mask_dir + f'mask_{ID}.npy'
            mask = torch.from_numpy(np.load(path_mask)).float().unsqueeze(0)
        
            if self.transform is not None:
                x, y, mask = self.transform(x, y, mask)
                return [ID, x, y, mask]
            
        elif self.y_dir is not None:
            path_y = self.y_dir + f'y_{ID}.npy'
            y = torch.from_numpy(np.load(path_y)).float().unsqueeze(0)
        
            if self.transform is not None:
                x, y, _ = self.transform(x, y)
                return [ID, x, y, None]
            
        else:
            if self.transform is not None:
                x, _, _ = self.transform(x)
            return [ID, x, None, None]

    

