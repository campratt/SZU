import numpy as np
import healpy as hp
#import torch
import os
from multiprocessing import Pool
import argparse

# Make small set of test data

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Hyperparameter Tuning")
    parser.add_argument('--data_dir', type=str, required=False, help='Data directory', default='./fullsky_data')
    parser.add_argument('--ID', type=str, required=False, help='Full-sky ID', default='00000')
    parser.add_argument('--output_dir', type=str, required=False, help='Directory to save outputs', default='./train_test_data/')
    parser.add_argument('--coords_path', type=str, required=False, help='Path to coordinate file', default='./coordinates.txt')
    parser.add_argument('--mask_path', type=str, required=False, help='Path to mask file', default='./mask.fits')
    parser.add_argument('--cores', type=int, required=False, help='Number of cores for parallel processing', default=1)
    args = parser.parse_args()
    return args




args = parse_args()
data_dir = os.path.join(args.data_dir)
output_dir = args.output_dir
ID = args.ID

N = 700 # number of samples per full-sky map
m = int(args.ID) # map ID to integer


# Frequency data with SZ signals injected based on instrumental beams
I857 = hp.read_map(os.path.join(data_dir,f'857GHz_{ID}.fits'))
I545 = hp.read_map(os.path.join(data_dir,f'545GHz_{ID}.fits'))
I353 = hp.read_map(os.path.join(data_dir,f'353GHz_{ID}.fits'))
I217 = hp.read_map(os.path.join(data_dir,f'217GHz_{ID}.fits'))
I143 = hp.read_map(os.path.join(data_dir,f'143GHz_{ID}.fits'))
I100 = hp.read_map(os.path.join(data_dir,f'100GHz_{ID}.fits'))
I70 = hp.read_map(os.path.join(data_dir,f'70GHz_{ID}.fits'))
I44 = hp.read_map(os.path.join(data_dir,f'44GHz_{ID}.fits'))
I30 = hp.read_map(os.path.join(data_dir,f'30GHz_{ID}.fits'))
maps = np.array([I30,I44,I70,I100,I143,I217,I353,I545,I857])

# SZ map smoothed to 10 arcminute FWHM
sz_map = hp.read_map(os.path.join(data_dir,f'sz_y_{ID}.fits'))

# Galaxy cluster mask
mask_map = hp.read_map(args.mask_path) 

# Coordinates
_, _, glons, glats = np.genfromtxt(args.coords_path).T

os.makedirs(os.path.join(output_dir, 'x/'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'y/'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'mask/'), exist_ok=True)


def func(n,m=m,N=N):
    count = m*N + n
    print(count)

    # frequency data
    data = []
    for i in range(len(maps)):
        data.append(hp.gnomview(maps[i], rot=(glons[n], glats[n]), xsize=512, reso=2, return_projected_map=True,no_plot=True))       
    np.save(os.path.join(output_dir,'x/') + f'x_{count}.npy', np.array(data))

    # SZ data
    data = hp.gnomview(sz_map, rot=(glons[n], glats[n]), xsize=512, reso=2, return_projected_map=True,no_plot=True)      
    np.save(os.path.join(output_dir, 'y/') + f'y_{count}.npy', np.array(data))

    # Mask data
    data = hp.gnomview(mask_map, rot=(glons[n], glats[n]), xsize=512, reso=2, return_projected_map=True,no_plot=True)      
    np.save(os.path.join(output_dir,'mask/') + f'mask_{count}.npy', np.array(data))



p = Pool(int(args.cores))
p.map(func, range(int(N)))

