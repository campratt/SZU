import numpy as np
import healpy as hp
#import torch
import os
from multiprocessing import Pool
import argparse

# Make small set of test data

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Hyperparameter Tuning")
    parser.add_argument('--data_dir', type=str, required=False, help='Data directory', default='./data')
    parser.add_argument('--map_id', type=str, required=False, help='Full-sky ID', default='00000')
    parser.add_argument('--output_dir', type=str, required=False, help='Directory to save outputs', default='./train_test_data/')
    parser.add_argument('--coords_path', type=str, required=False, help='Path to coordinate file', default='./coordinates.txt')
    parser.add_argument('--mask_path', type=str, required=False, help='Path to coordinate file', default='./coordinates.txt')
    parser.add_argument('--cores', type=int, required=False, help='Number of cores for parallel processing', default=1)


    args = parser.parse_args()
    return args


#def main():

args = parse_args()
d_simreal = os.path.join(args.data_dir, args.map_id)
d_save = args.output_dir

N = 700 # number of samples per full-sky map
m = int(args.map_id) # map ID to integer


# Frequency data with SZ signals injected based on instrumental beams
I857 = hp.read_map(os.path.join(d_simreal,'857GHz.fits'))
I545 = hp.read_map(os.path.join(d_simreal,'545GHz.fits'))
I353 = hp.read_map(os.path.join(d_simreal,'353GHz.fits'))
I217 = hp.read_map(os.path.join(d_simreal,'217GHz.fits'))
I143 = hp.read_map(os.path.join(d_simreal,'143GHz.fits'))
I100 = hp.read_map(os.path.join(d_simreal,'100GHz.fits'))
I70 = hp.read_map(os.path.join(d_simreal,'70GHz.fits'))
I44 = hp.read_map(os.path.join(d_simreal,'44GHz.fits'))
I30 = hp.read_map(os.path.join(d_simreal,'30GHz.fits'))
maps = np.array([I30,I44,I70,I100,I143,I217,I353,I545,I857])

# SZ map smoothed to 10 arcminute FWHM
sz_map = hp.read_map(os.path.join(d_simreal,'sz_y.fits') )

# Galaxy cluster mask
mask_map = hp.read_map(args.mask_path) 


_, _, glons, glats = np.genfromtxt(args.coords_path).T

os.makedirs(os.path.join(d_save, 'x_simreal/'), exist_ok=True)
os.makedirs(os.path.join(d_save, 'y_simreal/'), exist_ok=True)
os.makedirs(os.path.join(d_save, 'mask_simreal/'), exist_ok=True)


def func(n,m=m,N=N):
    count = m*N + n
    print(count)

    # frequency data
    data = []
    for i in range(len(maps)):
        data.append(hp.gnomview(maps[i], rot=(glons[n], glats[n]), xsize=512, reso=2, return_projected_map=True,no_plot=True))       
    np.save(os.path.join(d_save,'x_simreal/') + f'x_{count}.npy', np.array(data))

    # SZ data
    data = hp.gnomview(sz_map, rot=(glons[n], glats[n]), xsize=512, reso=2, return_projected_map=True,no_plot=True)      
    np.save(os.path.join(d_save, 'y_simreal/') + f'y_{count}.npy', np.array(data))

    data = hp.gnomview(mask_map, rot=(glons[n], glats[n]), xsize=512, reso=2, return_projected_map=True,no_plot=True)      
    np.save(os.path.join(d_save,'mask_simreal/') + f'mask_{count}.npy', np.array(data))






p = Pool(int(args.cores))
p.map(func, range(int(N)))

    

#if __name__ == '__main__':
#    main()