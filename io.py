# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13

@author: RSJ

"""

import glob,os
import numpy as np
def load_image_aooct(location):
    """In the AO-OCT data, volumetric images are stored as folders full of
    .npy B-scan files.
    """
    files = glob.glob(os.path.join(location,'bscan*.npy'))
    files.sort()
    image = [np.load(f) for f in files]
    dtype = image[0].dtype
    image = np.array(image,dtype=dtype)
    return image

def load_image_clinical_org(location):
    """In the clinical org data, individual images are single .npy
    files, each representing a B-scan.
    """
    image = np.load(location)
    return image
