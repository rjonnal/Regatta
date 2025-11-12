# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 12:49:04 2025

@author: ZAQ
"""

import RegATTA as rg
import glob
import numpy as np

# reference_folder = 'path/to/reference/bscans'
reference_folder = r'E:\Registration\data_Ravi\bscans_aooct\00000'
# target_folder = 'path/to/target/bscans'
target_folder = r'E:\Registration\data_Ravi\bscans_aooct\00001'

# --- Load regatta classes ---
refim = rg.ReferenceImage(reference_folder)
ris = rg.RegisteredImageSeries()

# target data exist in numbered subfolders of mydata:
# target_folders = sorted(glob.glob('mydata/00*'))
target_folders = sorted(glob.glob(r'E:\Registration\data_Ravi\bscans_aooct\00*'))

for tf in target_folders: 
    ris.add(tf) # tf is a directory, might use preprocess.load_volumes to load images in ris.
    
    # ris knows how to add an image from a folder, using custom or default load_image function
    # when adding the image, ris doesn't load the data, it just stores the value of tf and
    # initializes dy, dz, dx, xc maps to zero
    # how does ris know dimensions of data? maybe initialize w/ dimensions, or load one test image
    target = ris.get(tf)

    # for rigid bodies (including serial B-scans)
    dy,dz,dx,xc = refim.register(target,poxc=False)
    ris.set_ymap(dy)
    ...

    # for volumes consiting of rigid single B-scans
    sy,sz,sx = target.shape
    for y in sy:
        bscan = target[y,:,:]
        dy,dz,dx,xc = refim.register(bscan)
        ris.set_ymap(dy,y=y)

# At this point we have filled all the maps in ris, so we can start averaging volumes
# Thus, ris needs to know how to return the dewarped/deformed data; it needs to determine
# a fixed size for all returned images
ris.finalize()

sum = ris.get_sum_image()
counter = ris.get_counter_image()

for tf in target_folder:
    target = ris.get(tf)
    temp = np.zeros(target.shape)
    temp[np.where(not np.isnan(target))] = 1
    sum = sum + target
    counter = counter + temp

av = sum/counter