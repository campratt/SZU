import numpy as np
import healpy as hp
import os
import argparse

from astropy.constants import c
sol = c.to('km/s').value

hplanck_cgs = 6.6261e-27 #cm2 g s-1
sol_cgs = c.to('cm/s').value
kb_cgs = 1.3807e-16 #cm2 g s-2 K-1
Tcmb = 2.7255

def gnu_Kcmb(nu_GHz,Tcmb=2.7255):
    h_over_k = hplanck_cgs/kb_cgs
    nu=nu_GHz*1e9
    x = h_over_k*nu/Tcmb
    return (x*(np.exp(x)+1)/(np.exp(x)-1)) - 4


def parse_args():
    parser = argparse.ArgumentParser(description="Retrieve Han21 data")
    parser.add_argument('--output_dir', type=str, required=False, help='Directory to save outputs', default='./Han21')
    parser.add_argument('--ID', type=str, required=False, help='Full-sky ID', default='00000')
    args = parser.parse_args()
    return args

args = parse_args()


ID = args.ID

save_dir = os.path.join(args.output_dir)
os.makedirs(save_dir, exist_ok=True)

d = 'https://portal.nersc.gov/project/cmb/data/generic/mmDL/healpix/%s/'%(ID)
    
url = os.path.join(d,'tsz_148ghz_%s.fits'%(ID))
fn_save = os.path.join(save_dir,'tsz_148ghz_%s.fits'%(ID))

# Download original Han21 data
os.system('wget -O %s %s'%(fn_save,url))

sz_148 = hp.read_map(fn_save)

os.system('rm %s'%fn_save)

#dat = fits.open(url,cache=False)

# Resize to NSIDE=2048
sz_148 = hp.alm2map(hp.map2alm(sz_148,lmax=6000),2048)
hp.write_map(fn_save,sz_148,overwrite=True)


# Convert to dimensionless Compton-y parameter
fn_sz_save = os.path.join(save_dir,'sz_y_%s.fits'%(ID))
sz = sz_148*1e-6/(gnu_Kcmb(148)*Tcmb) #dimensionless
sz[sz<0] = 0
hp.write_map(fn_sz_save,sz,overwrite=True)

