import numpy as np
import healpy as hp
from numba import njit
from multiprocessing import Pool
import os
from os import listdir
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Inject Han21 SZ maps in PR4 frequency data")
    parser.add_argument('--freq_dir', type=str, required=False, help='Directory to PR4 frequency maps', default='./freq_dir')
    parser.add_argument('--sz_dir', type=str, required=False, help='Directory to SZ maps', default='./Han21')
    parser.add_argument('--ID', type=str, required=False, help='Full-sky ID', default='00000')
    parser.add_argument('--output_dir', type=str, required=False, help='Directory to save outputs', default='./fullsky_data')
    parser.add_argument('--cores', type=int, required=False, help='Number of cores for multiprocessing', default=16)
    
    args = parser.parse_args()
    return args

args = parse_args()

@njit
def ang_dist(Ra1, Dec1, Ra2, Dec2):
    dec1 = np.pi/2. - Dec1/180.*np.pi
    dec2 = np.pi/2. - Dec2/180.*np.pi
    
    cosad = np.cos(dec1) * np.cos(dec2) + np.sin(dec1)* np.sin(dec2) * np.cos((Ra1 - Ra2)/180.*np.pi)
    ad = np.arccos(cosad)
    return ad * 180. / np.pi



gnu_freq = {23 :-5.371,
            30 :-5.336,
            33 :-5.291,
            41 :-5.212,
            44 :-5.178,
            61 :-4.933,
            70 :-4.766,
            94 :-4.261,
            100:-4.031,
            143:-2.785,
            217: 0.187,
            353: 6.205,
            545:14.455,
            857:26.335}

res_freq = {23 :52.8,
            30 :33.16,
            33 :39.6,
            41 :30.6,
            44 :28.09,
            61 :21,
            70 :13.08,
            94 :13.2,
            100:9.59,
            143:7.18,
            217: 4.87,
            353: 4.7,
            545:4.73,
            857:4.51}

# Make frequency maps

ID = args.ID
output_dir = args.output_dir
freq_dir = args.freq_dir
sz_dir = args.sz_dir
cores = args.cores

os.makedirs(output_dir,exist_ok=True)



for fn in sorted(listdir(freq_dir)):
    
    ### Planck
    if '30GHz'in fn or '30-field' in fn:
        I30 = hp.read_map(os.path.join(freq_dir,fn))
    elif '44GHz' in fn or '44-field' in fn:
        I44 = hp.read_map(os.path.join(freq_dir,fn))
    elif '70GHz' in fn or '70-field' in fn:
        I70 = hp.read_map(os.path.join(freq_dir,fn))
    elif '100GHz' in fn or '100-field' in fn:
        I100 = hp.read_map(os.path.join(freq_dir,fn))
    elif '143GHz' in fn or '143-field' in fn:
        I143 = hp.read_map(os.path.join(freq_dir,fn))
    elif '217GHz' in fn or '217-field' in fn:
        I217 = hp.read_map(os.path.join(freq_dir,fn))
    elif '353GHz' in fn or '353-field' in fn:
        I353 = hp.read_map(os.path.join(freq_dir,fn))
    elif '545GHz' in fn or '545-field' in fn:
        I545 = hp.read_map(os.path.join(freq_dir,fn))
    elif '857GHz'in fn or '857-field' in fn:
        I857 = hp.read_map(os.path.join(freq_dir,fn))
        


if hp.get_nside(I30)!=2048:     
    l2048, b2048 = hp.pix2ang(2048, np.arange(2048**2*12), lonlat=True)
    indices_1024 = hp.ang2pix(1024, l2048, b2048, lonlat=True)
    I30_2048 = I30[indices_1024]
    I44_2048 = I44[indices_1024]
    I70_2048 = I70[indices_1024]

    I30 = I30_2048
    I44 = I44_2048
    I70 = I70_2048

freqs_planck = [30,44,70,100,143,217,353,545,857]
maps_planck = [I30, I44, I70, I100, I143, I217, I353, I545, I857]






# Smooth SZ map to 10 arcmin resolution
sz_map = hp.read_map(os.path.join(sz_dir,f'sz_y_{ID}.fits'))

fwhm_arc = 10
nside = 2048
m = hp.ud_grade(sz_map.copy(),nside)
rext_sigma = 5

la, ba = hp.pix2ang(nside, np.arange(nside**2*12), lonlat=True)
pix = np.arange(12*nside**2)

def convolve_gauss(idx):
    fwhm = np.radians(fwhm_arc / 60)
    l,b = hp.pix2ang(nside, idx, lonlat=True)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    radius_calc = rext_sigma*sigma
    neighbors = hp.query_disc(nside, hp.pix2vec(nside, idx), radius_calc)
    lt, bt = la[neighbors], ba[neighbors]
    angt = np.nan_to_num(ang_dist(l, b, lt, bt))*np.pi/180
    beam = np.exp(-(angt/sigma)**2)
    val = np.sum(beam*m[neighbors])/np.sum(beam)
    return val


p = Pool(cores)
sz = p.map(convolve_gauss,pix)
sz = np.array(sz).flatten()
hp.write_map(os.path.join(output_dir,f'sz_y_{ID}.fits'),sz,overwrite=True)


# Inject SZ signal in Planck frequency maps and smooth to beam resolution
for i, freq in enumerate(freqs_planck):
    
    fwhm_arc = res_freq[freq]

    nsides = [128,256,512,1024,2048]
    for nside in nsides:
        if fwhm_arc/hp.nside2resol(nside, arcmin=True)>5:
            break

    m = hp.ud_grade(sz_map.copy(),nside)
    rext_sigma = 5


    la, ba = hp.pix2ang(nside, np.arange(nside**2*12), lonlat=True)
    pix = np.arange(12*nside**2)



    def convolve_gauss(idx):
        fwhm = np.radians(fwhm_arc / 60)
        l,b = hp.pix2ang(nside, idx, lonlat=True)
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        radius_calc = rext_sigma*sigma
        neighbors = hp.query_disc(nside, hp.pix2vec(nside, idx), radius_calc)
        lt, bt = la[neighbors], ba[neighbors]
        angt = np.nan_to_num(ang_dist(l, b, lt, bt))*np.pi/180
        beam = np.exp(-(angt/sigma)**2)
        val = np.sum(beam*m[neighbors])/np.sum(beam)
        return val

    
    #cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    cores = args.cores
    p = Pool(cores)

    # Smooth SZ signal
    sz = p.map(convolve_gauss,pix)
    sz = np.array(sz).flatten()
    sz = hp.ud_grade(sz,2048)
    #hp.write_map(os.path.join(output_dir,f'sz_{freqs_planck[i]}GHz_{ID}.fits'),sz,overwrite=True)

    # Inject SZ signal
    freq_map = maps_planck[i] + (gnu_freq[freqs_planck[i]]*sz)
    hp.write_map(os.path.join(output_dir,f'{freqs_planck[i]}GHz_{ID}.fits'),freq_map,overwrite=True)
