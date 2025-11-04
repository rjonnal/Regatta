# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 13:30:30 2025

Generate mask for interested retina layers

@author: ZAQ
"""
#%% ===================================== I/O ===================================================
save_mask = True
save_masked_img = True
bscan_idx = 50
threshold = 2000
display_max = 6000

#%% ===================================== Module Import ===================================================
#FILE PATHS AND NAMES
from tkinter import Tk
from tkinter import filedialog
import numpy as np
import functions as blobf
import config as cfg
import glob, os
import matplotlib.pyplot as plt
from pathlib import Path

#%% Read, crop, and display Slow images (From Clinical OCT)
def choose_folder():
    # Create a hidden Tkinter window
    root = Tk()
    root.withdraw()  # Hide the root window    
    root.attributes('-topmost', True)  # Bring the dialog to the front
    root.update()
    # Open the folder selection dialog
    folder_path = filedialog.askdirectory(title="Select a Folder")
    # Destroy the root window to free resources
    root.destroy()
    if folder_path:
        print(f"Selected folder: {folder_path}")
    else:
        print("No folder selected.")
    
    return folder_path

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
    bscan_z1 = np.min(points) - 70 #cfg.z_padding_px
    bscan_z2 = np.min(points) + 70
    return bscan_z1, bscan_z2

#%% Generate a mask
def generate_mask_slice(slice_img, threshold, bscan_idx, mask_orig_img, plot, save_output):
    #FIXME
    mask = np.abs(slice_img) > threshold
    # mask = slice_img > threshold
    mask_bin = mask.astype(np.uint8)   # 0 and 1
    
    if mask_orig_img == True:
        masked_img = np.where(mask, slice_img, 0)
    
    if plot == True:
        plt.figure()
        plt.imshow(mask, cmap="gray", aspect='auto', vmax = 0.5)
        plt.title(f"Mask of Bscan {bscan_idx}")
        if save_output == True:
            plt.save(f"Bscan_{bscan_idx}_mask.png")
        
        plt.figure()
        plt.imshow(np.abs(masked_img), cmap="gray", aspect='auto', vmax = 1200)
        plt.title(f"Masked Bscan {bscan_idx}")
        plt.show()
        if save_output == True:
            plt.save(f"Bscan_{bscan_idx}_masked.png")
            
    if save_output == True:
        np.save(f"Bscan_{bscan_idx}_mask.npy", mask)
        np.save("Masked_Bscan_{bscan_idx}.npy", masked_img)
    
    if mask_orig_img == True:
        return mask_bin, masked_img
    else:
        return mask_bin

def generate_mask_volume(vol, threshold, mask_orig_img, plot, save_output):
    dy, dz, dx = np.shape(vol)
    
    # mask_vol = np.zeros_like([dy, dz, dx])
    mask_vol = []
    
    if mask_orig_img == True:
        # masked_vol = np.zeros_like([dy, dz, dx])
        masked_vol = []
        for bscan_idx in range(0, dy):
            mask_vol_tmp, masked_vol_tmp = generate_mask_slice(vol[bscan_idx,:,:],
                                                                       threshold,
                                                                       bscan_idx,
                                                                       mask_orig_img,
                                                                       plot,
                                                                       False)
            mask_vol.append(mask_vol_tmp)
            masked_vol.append(masked_vol_tmp)
        if save_output == True:
            np.save("masked_volume.npy", masked_vol)
        return mask_vol, masked_vol
    else:
        for bscan_idx in range(0, dy):
            mask_vol_tmp = generate_mask_slice(vol[bscan_idx,:,:],
                                               threshold,
                                               bscan_idx,
                                               mask_orig_img,
                                               plot,
                                               save_output)
            mask_vol.append(mask_vol_tmp)
        return mask_vol
    

#%% Main Block
def main():
    # Volume path
    ref_volume_folder = Path(choose_folder()) #/Path/Slow/15_24_39
    # ref_bscan_folder = r".\..\3_RVS_real_data\Slow\15_24_39_bscans _orig"
    tar_volume_folder = Path(choose_folder()) #/Path/Slow/15_24_57
    # tar_bscan_folder = r".\..\3_RVS_real_data\Slow\15_24_57_bscans _orig"
    
    # Volume name
    ref_vol_name = ref_volume_folder.parts[-1] #/Path/Slow/15_24_39
    print(f"Reference Volume is {ref_vol_name}")
    tar_vol_name = tar_volume_folder.parts[-1] #/Path/Slow/15_24_39
    print(f"Target Volume is {tar_vol_name}")
    
    #### Read the original images ####
    print("------- Reading original Images ------")
    ref_bscan_folder = f"{ref_volume_folder}/{ref_vol_name}_bscans"#/Path/Slow/15_24_39/15_24_39_bscans
    tar_bscan_folder = f"{tar_volume_folder}/{tar_vol_name}_bscans"#/Path/Slow/15_24_57/15_24_57_bscans
    
    ref_img = get_volume(ref_bscan_folder)
    tar_img = get_volume(tar_bscan_folder)
    
    # Show raw image #
    display_range = np.arange(0,display_max)
    fig, ax = plt.subplots(1, 2)
    
    ax[0].imshow(np.abs(ref_img[1,:,:]), cmap="gray", aspect='auto', clim=np.percentile(display_range,(1,99)))
    ax[0].set_title("reference image")
    
    im2 = ax[1].imshow(np.abs(tar_img[1,:,:]), cmap="gray", aspect='auto', clim=np.percentile(display_range,(1,99)))
    ax[1].set_title("target image")
    
    fig.colorbar(im2, ax=ax, orientation="vertical", fraction=0.05, pad=0.04)
    
    # Crop depth #
    ref_z1, ref_z2 = auto_crop(ref_img)
    ref_img = ref_img[:,ref_z1:ref_z2,:]
    tar_z1, tar_z2 = auto_crop(tar_img)
    tar_img = tar_img[:,tar_z1:tar_z2,:]
    
    sy, sz, sx = ref_img.shape
    
    # Show cropped images #
    fig, ax = plt.subplots(1, 2)
    
    ax[0].imshow(np.abs(ref_img[1,:,:]), cmap="gray", aspect='auto', clim=np.percentile(display_range,(1,99)))
    ax[0].set_title("cropped reference image")
    
    im2 = ax[1].imshow(np.abs(tar_img[1,:,:]), cmap="gray", aspect='auto', clim=np.percentile(display_range,(1,99)))
    ax[1].set_title("cropped target image")
    
    fig.colorbar(im2, ax=ax, orientation="vertical", fraction=0.05, pad=0.04)
    
    # Show en face images #
    # ref img
    z_projection = np.mean(np.abs(ref_img),axis=1)           
    print('projection size: ' + str(z_projection.shape)) 
    fig = plt.figure(figsize=(10,10*cfg.bscans_per_volume/172))
    ax = fig.add_axes([0,0,1,1])
    ax.set_xticks([]) # comment out to show axes
    ax.set_yticks([])
    # ax.set_title('depth range %s - %s'%(projection_z1,projection_z2),fontsize=30)
    ax.set_title('ref enface image',fontsize=30)
    ax.imshow(z_projection,aspect='auto',clim=np.percentile(z_projection,(1,99)),cmap='gray')
    
    # tar img
    z_projection = np.mean(np.abs(tar_img),axis=1)           
    print('projection size: ' + str(z_projection.shape)) 
    fig = plt.figure(figsize=(10,10*cfg.bscans_per_volume/172))
    ax = fig.add_axes([0,0,1,1])
    ax.set_xticks([]) # comment out to show axes
    ax.set_yticks([])
    # ax.set_title('depth range %s - %s'%(projection_z1,projection_z2),fontsize=30)
    ax.set_title('tar enface image',fontsize=30)
    ax.imshow(z_projection,aspect='auto',clim=np.percentile(z_projection,(1,99)),cmap='gray')
    bscan = tar_img[bscan_idx,:,:]
    bscan_mask, masked_bscan = generate_mask_slice(bscan, threshold, bscan_idx, mask_orig_img=True, plot=False, save_output=False)
    
    ref_vol_mask, ref_vol_masked = generate_mask_volume(ref_img, threshold, mask_orig_img=True, plot=False, save_output=False)
    tar_vol_mask, tar_vol_masked = generate_mask_volume(tar_img, threshold, mask_orig_img=True, plot=False, save_output=False)
    
    # np.save("bscan_50_masked.npy", masked_bscan)
    if save_mask == True:
        np.save(f"{ref_volume_folder}/{ref_vol_name}_mask.npy", ref_vol_mask)
        np.save(f"{tar_volume_folder}/{tar_vol_name}_mask.npy", ref_vol_mask)
    if save_masked_img == True:
        np.save(f"{ref_volume_folder}/{ref_vol_name}_masked.npy", ref_vol_masked)
        np.save(f"{tar_volume_folder}/{tar_vol_name}_masked.npy", tar_vol_masked)

if __name__ == "__main__":
    main()

