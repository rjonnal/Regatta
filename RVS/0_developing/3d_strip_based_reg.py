# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 17:09:19 2025

3D windowing Strip based registration

@author: ZAQ
"""
show = True
sigma = 5
reference_index = 0
#%% ===================================== Module Import ===================================================
#FILE PATHS AND NAMES
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter import filedialog
import numpy as np
import functions as blobf
import config as cfg
import glob, os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path
import time
import sys

#%% Functions
def get_volume(bscan_folder):
    flist = glob.glob(os.path.join(bscan_folder,'complex*.npy'))
    flist.sort()
    vol = [np.load(f) for f in flist]
    vol = np.array(vol)
    return vol

def auto_crop(img):
    zprof = np.nanmean(img,axis=(0,2))
    plt.plot(zprof)
    points = blobf.get_peaks_ui(zprof)
    bscan_z1 = np.min(points) - 45 #cfg.z_padding_px
    bscan_z2 = np.min(points) + 32
    return bscan_z1, bscan_z2

#%% Load images
ref_fn = r"C:\Users\rjonn\OneDrive - UC Davis Health\CHOIR\5_Projects\2_Registration\1_registration_pipeline\1_RVS\3_RVS_real_data\Slow\15_24_39"
# tar_fn = r"C:\Users\rjonn\OneDrive - UC Davis Health\CHOIR\5_Projects\2_Registration\1_registration_pipeline\1_RVS\3_RVS_real_data\Slow\15_24_39"
tar_fn = r"C:\Users\rjonn\OneDrive - UC Davis Health\CHOIR\5_Projects\2_Registration\1_registration_pipeline\1_RVS\3_RVS_real_data\Slow\15_24_57"

ref_filename = "15_24_39"
# tar_filename = "15_24_39"
tar_filename = "15_24_57"

ref_bscan_folder = r"C:\Users\rjonn\OneDrive - UC Davis Health\CHOIR\5_Projects\2_Registration\1_registration_pipeline\1_RVS\3_RVS_real_data\Slow\15_24_39\15_24_39_bscans"
# tar_bscan_folder = r"C:\Users\rjonn\OneDrive - UC Davis Health\CHOIR\5_Projects\2_Registration\1_registration_pipeline\1_RVS\3_RVS_real_data\Slow\15_24_39\15_24_39_bscans"
tar_bscan_folder = r"C:\Users\rjonn\OneDrive - UC Davis Health\CHOIR\5_Projects\2_Registration\1_registration_pipeline\1_RVS\3_RVS_real_data\Slow\15_24_57\15_24_57_bscans"

ref_img = get_volume(ref_bscan_folder)
tar_img = get_volume(tar_bscan_folder)

# Crop depth #
ref_z1, ref_z2 = auto_crop(ref_img)
ref_img = ref_img[:,ref_z1:ref_z2,:]
tar_z1, tar_z2 = auto_crop(tar_img)
tar_img = tar_img[:,tar_z1:tar_z2,:]

# Show one bscan
plt.figure()
plt.imshow(np.abs(ref_img[50,:,:]), clim=np.percentile(np.abs(ref_img),(1,99)), aspect='auto', cmap='gray')
plt.title("1 ref bscan")

plt.figure()
plt.imshow(np.abs(tar_img[50,:,:]), clim=np.percentile(np.abs(tar_img),(1,99)), aspect='auto',cmap='gray')
plt.title("1 tar bscan")

# Show en face images #
# ref img
ref = np.mean(np.abs(ref_img),axis=1) # z projection ref    
print('projection size: ' + str(ref.shape)) 
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.set_xticks([]) # comment out to show axes
ax.set_yticks([])
# ax.set_title('depth range %s - %s'%(projection_z1,projection_z2),fontsize=30)
ax.set_title('ref enface image',fontsize=30)
ax.imshow(ref,aspect='auto',clim=np.percentile(ref,(1,99)),cmap='gray')

# tar img
tar = np.mean(np.abs(tar_img),axis=1) # z projection tar       
print('projection size: ' + str(tar.shape)) 
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.set_xticks([]) # comment out to show axes
ax.set_yticks([])
# ax.set_title('depth range %s - %s'%(projection_z1,projection_z2),fontsize=30)
ax.set_title('tar enface image',fontsize=30)
ax.imshow(tar,aspect='auto',clim=np.percentile(tar,(1,99)),cmap='gray')

#%% preprocess image
# Add preprocessing to normalize intensities
def preprocess_volume(vol):
    """Normalize and enhance volume for better correlation"""
    # Remove outliers and normalize
    vol = np.clip(vol, np.percentile(vol, 1), np.percentile(vol, 99))
    vol = (vol - np.mean(vol)) / (np.std(vol) + 1e-12)
    return vol

# Apply before registration
ref_img = preprocess_volume(ref_img)
tar_img = preprocess_volume(tar_img)
#%% Test 2D
clim = np.percentile(ref,(0.5,99.9))

#precompute ref fft
fref = np.fft.fft2(ref)
afref = np.abs(fref)

output_header = ['xshift','yshift','ycoord','correlation','reference_index','reference_filename']
output_rows = []

output_filename = r"C:\Users\rjonn\OneDrive - UC Davis Health\CHOIR\5_Projects\2_Registration\1_registration_pipeline\1_RVS\3_RVS_real_data\Slow\strip_registration.csv"
sy,sx = tar.shape
output_rows = []
print('Start Registering.')
for row in range(sy):
    if not show:
        print('.',end='')
        sys.stdout.flush()
    y = np.arange(sy)-row
    g = np.exp(-(y**2)/(2*sigma**2))
    strip = (g*tar.T).T # broadcast strip; Similar with strip = tar * g[:, None]
    fstrip = np.conj(np.fft.fft2(strip))
    afstrip = np.abs(fstrip)
    
    xc = np.abs(np.fft.ifft2(fref*fstrip/afref/afstrip))
    py,px = np.unravel_index(np.argmax(xc),xc.shape)
    xcmax = np.max(xc)
    yshift,xshift = py,px
    
    # Wraparound bc shifts larger than half of the array size represent negative displacements due to periodicity of FFT
    if yshift>sy//2:
        yshift = yshift - sy
    if xshift>sx//2:
        xshift = xshift - sx
        
    xplot = np.array([0,sx,sx,0,0])
    yplot = np.array([row-2*sigma,row-2*sigma,row+2*sigma,row+2*sigma,row-2*sigma])

    rxplot = xplot+xshift
    ryplot = yplot+yshift

    # we want to record the coordinates in the reference frame where this strip goes
    # the x location is given by xshift
    # the y location is given by row+yshift
    output_rows.append([xshift,yshift,row+yshift,xcmax,reference_index,ref_filename])
    if show:
        fig_save_dir = r"C:\Users\rjonn\OneDrive - UC Davis Health\CHOIR\5_Projects\2_Registration\1_registration_pipeline\1_RVS\3_RVS_real_data\Slow\2d_strip_based"
        plt.clf()
        plt.subplot(2,3,1)
        plt.imshow(ref,cmap='gray',clim=clim)
        plt.plot(rxplot,ryplot,'y-')
        plt.title('ref')
        plt.xlim((0,sx))
        plt.ylim((sy,0))
        plt.subplot(2,3,4)                
        plt.imshow(tar,cmap='gray',clim=clim)
        plt.title('target')
        plt.plot(xplot,yplot,'y-')
        plt.xlim((0,sx))
        plt.ylim((sy,0))
        plt.subplot(2,3,3)
        plt.imshow(strip,cmap='gray',clim=clim)
        plt.title('smoothed strip')
        plt.subplot(2,3,6)
        plt.imshow(20*np.log10(np.abs(xc)))
        plt.plot(px,py,'rs',markersize=10,markerfacecolor='none')
        plt.title('xcorr map')
        plt.subplot(2,3,2)
        plt.title('zoom ref')
        plt.imshow(ref,cmap='gray',clim=clim,aspect='auto')
        plt.plot(rxplot,ryplot,'y-')
        plt.xlim(rxplot[:2])
        plt.ylim(ryplot[-2:])
        plt.subplot(2,3,5)
        plt.title('zoom target')
        plt.imshow(tar,cmap='gray',clim=clim,aspect='auto')
        plt.plot(xplot,yplot,'y-')
        plt.xlim(xplot[:2])
        plt.ylim(yplot[-2:])
        plt.suptitle('yshift,xshift=%d, %d'%(yshift,xshift))
        plt.tight_layout()
        # plt.savefig(f"{fig_save_dir}/bscan_{row}_shifts.png")
        plt.pause(.000001)
    
# df = pd.DataFrame(output_rows,columns=output_header)
# df.to_csv(output_filename)
# print()

#%% 3D strip based
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

def prepare_reference_fft(ref_vol):
    """Precompute FFT(ref) and |FFT(ref)| once."""
    fref = np.fft.fftn(ref_vol)
    afref = np.abs(fref)
    return fref, afref

def register_volume_by_y_slabs(
    tar_vol, ref_vol, sigma_y, show=False,
    clim=None, save_dir=None, reference_index=None, ref_filename=None
):
    """
    3D analogue of 2D per-row strip registration.
    Axis order must be (y, z, x).

    For each row (y), window tar_vol with a Gaussian slab along y,
    do 3D phase correlation to ref_vol, find (dy, dz, dx), and log results.

    Returns:
        output_rows: list of [dx, dy, dz, row+dy, xcmax, reference_index, ref_filename]
        shifts_xyz:  (sy, 3) array of [dy, dz, dx] per row
    """
    sy, sz, sx = tar_vol.shape
    fref, afref = prepare_reference_fft(ref_vol)
    output_rows = []
    shifts_xyz = np.zeros((sy, 3), dtype=int)

    # simple default display limits for quick plots
    if clim is None:
        vmin = np.percentile(ref_vol, 1)
        vmax = np.percentile(ref_vol, 99)
        clim = (vmin, vmax)

    # precompute XY polygon for plotting on MIP(y,x)
    xplot = np.array([0, sx, sx, 0, 0], dtype=float)
    eps = 1e-12
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for row in range(sy):
        if not show:
            print('.', end='', flush=True)

        # ----- 1) Gaussian slab (along y only), broadcast over z,x -----
        y = np.arange(sy) - row
        g = np.exp(-(y**2) / (2.0 * sigma_y**2))     # (sy,)
        slab = tar_vol * g[:, None, None]            # (sy, sz, sx)
        
        # # ----- 1) Rectangular slab (along y only), broadcast over z,x -----
        # y = np.arange(sy)   # 0 ... sy-1
        # center = row        # center of the slab
        # half_width = np.int32(np.floor(sigma_y // 2))  # slab_width in voxels, e.g. 5
        
        # # rectangle function: 1 inside the window, 0 outside
        # rect = np.zeros_like(y, dtype=float)
        # y1, y2 = max(0, center - half_width), min(sy, center + half_width + 1)
        # rect[y1:y2] = 1
        
        # # broadcast over z,x
        # slab = tar_vol * rect[:, None, None]      # (sy, sz, sx)
        # # slab = tar_vol[y1:y2,:,:]
        # slab_ref = ref_vol * rect[:, None, None]  # (sy, sz, sx)

        
        # np.save(rf"E:\Registration\Slow\3d_strip_based_debug\target_row_{row}_sigma_{sigma_y}\target_slab_row_{row}.npy", slab)
        
        # for iy in np.arange(0,np.int32(sigma_y)):
        #     bscan_idx = center-half_width+iy
        #     plt.figure()
        #     plt.imshow(np.abs(slab[iy,:,:]), cmap='gray', aspect='auto', clim=clim)
        #     plt.title(f"target image bscan ({bscan_idx},:,:)")
        #     plt.tight_layout()
        #     os.makedirs(rf"E:\Registration\Slow\3d_strip_based_debug\target_row_{row}_sigma_{sigma_y}\target_slab_bscans", exist_ok=True)
        #     plt.savefig(rf"E:\Registration\Slow\3d_strip_based_debug\target_row_{row}_sigma_{sigma_y}\target_slab_bscans\target_slice_{bscan_idx}.png")
            

        # for iz in np.arange(0,sz):
        #     plt.figure(figsize=(10,1))
        #     plt.imshow(np.abs(slab[:,iz,:]), cmap='gray', aspect='auto', clim=clim)
        #     plt.title(f"target enface (:,{iz},:)")
        #     plt.tight_layout()
        #     os.makedirs(rf"E:\Registration\Slow\3d_strip_based_debug\target_row_{row}_sigma_{sigma_y}\target_slab_enface", exist_ok=True)
        #     plt.savefig(rf"E:\Registration\Slow\3d_strip_based_debug\target_row_{row}_sigma_{sigma_y}\target_slab_enface\target_enface_{iz}.png")
            
            
            
        # ----- 2) FFT(slab), conj for correlation, magnitude for normalization -----
        fslab = np.conj(np.fft.fftn(slab))
        afslab = np.abs(fslab)

        # ----- 3) Phase correlation (3D) -----
        xc = np.abs(np.fft.ifftn((fref * fslab) / ((afref + eps) * (afslab + eps))))

        # ----- 4) Peak + unwrap to signed shifts -----
        py, pz, px = np.unravel_index(np.argmax(xc), xc.shape)
        xcmax = float(xc[py, pz, px])

        dy, dz, dx = int(py), int(pz), int(px)
        if dy > sy // 2: dy -= sy
        if dz > sz // 2: dz -= sz
        if dx > sx // 2: dx -= sx

        shifts_xyz[row] = (dy, dz, dx)

        # ----- 5) For plotting: rectangle (y extent ~ row ± 2σ) projected to (y,x) MIP -----
        yplot = np.array([row - 2 * sigma_y, row - 2 * sigma_y,
                          row + 2 * sigma_y, row + 2 * sigma_y,
                          row - 2 * sigma_y], dtype=float)

        # where that rectangle lands in the reference, after shift
        rxplot = xplot + dx
        ryplot = yplot + dy

        # ----- 6) Save a concise row result (aligns with your 2D output shape) -----
        output_rows.append([
            dx, dy, dz,                 # shifts (note: added dz for 3D)
            int(row + dy),              # row_in_ref (center line of slab)
            xcmax,
            reference_index,
            ref_filename
        ])

        # ----- 7) Optional visualization (quick and analogous) -----
        
        if show:
            # En face MIPs (y,x) to mimic your 2D overview quickly
            ref_mip_yx = np.max(ref_vol, axis=1)   # max over z  -> (y, x)
            tar_mip_yx = np.max(tar_vol, axis=1)

            # A representative z-slice for the strip visualization (choose mid-z)
            zc = sz // 2
            strip_yx = slab[:, zc, :]

            plt.clf()
            plt.subplot(2, 3, 1)
            plt.imshow(ref_mip_yx, cmap='gray', clim=clim, aspect='auto')
            plt.plot(rxplot, ryplot, 'y-')
            plt.title('ref MIP (y,x)')
            plt.xlim((0, sx)); plt.ylim((sy, 0))

            plt.subplot(2, 3, 4)
            plt.imshow(tar_mip_yx, cmap='gray', clim=clim, aspect='auto')
            plt.plot(xplot, yplot, 'y-')
            plt.title('tar MIP (y,x)')
            plt.xlim((0, sx)); plt.ylim((sy, 0))

            plt.subplot(2, 3, 3)
            plt.imshow(strip_yx, cmap='gray', clim=clim, aspect='auto')
            plt.title('slab @ z≈mid (y,x)')

            plt.subplot(2, 3, 6)
            # show a z-mid slice of correlation OR a MIP; here: MIP over z for visibility
            xc_mip_yx = np.max(xc, axis=1)  # (y,x)
            plt.imshow(20 * np.log10(np.maximum(xc_mip_yx, 1e-12)), aspect='auto')
            plt.plot(px, py, 'rs', markersize=8, markerfacecolor='none')
            plt.title('xcorr MIP (y,x)')

            plt.subplot(2, 3, 2)
            plt.title('zoom ref MIP')
            plt.imshow(ref_mip_yx, cmap='gray', clim=clim, aspect='auto')
            plt.plot(rxplot, ryplot, 'y-')
            plt.xlim(rxplot[:2]); plt.ylim(ryplot[-2:])

            plt.subplot(2, 3, 5)
            plt.title('zoom tar MIP')
            plt.imshow(tar_mip_yx, cmap='gray', clim=clim, aspect='auto')
            plt.plot(xplot, yplot, 'y-')
            plt.xlim(xplot[:2]); plt.ylim(yplot[-2:])

            plt.suptitle(f'dy, dz, dx = {dy}, {dz}, {dx}   |   xcmax={xcmax:.3g}')
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, f"row_{row:04d}_shifts.png"), dpi=120)
            plt.pause(1e-6)
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, f"row_{row:04d}_shifts.png"), dpi=120)
    if not show:
        print()  # newline after progress dots
    return output_rows, shifts_xyz

start = time.perf_counter()   # start timer
# ref_vol, tar_vol are shape (sy, sz, sx), e.g. (600, 120, 512)
sigma_y = 7
# out_rows, shifts = register_volume_by_y_slabs(
#     tar_img, ref_img, sigma_y,
#     show=True,                          # set False to speed up
#     clim=(np.percentile(np.abs(ref_img),1), np.percentile(np.abs(ref_img),99)),
#     save_dir=r"C:\Users\rjonn\OneDrive - UC Davis Health\CHOIR\5_Projects\2_Registration\1_registration_pipeline\1_RVS\3_RVS_real_data\Slow\3d_strip_based",
#     reference_index=0,
#     ref_filename="ref_vol.bin"
# )
out_rows, shifts = register_volume_by_y_slabs(
    tar_img, ref_img, sigma_y,
    show=True,                          # set False to speed up
    clim=(np.percentile(np.abs(ref_img),1), np.percentile(np.abs(ref_img),99)),
    save_dir=False,
    reference_index=0,
    ref_filename="ref_vol.bin"
)
time.sleep(2)  # simulate work

end = time.perf_counter()
print(f"Elapsed time: {end - start:.2f} seconds")
