# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 11:42:26 2025

@author: ZAQ
"""
from .preprocess import Preprocess
import numpy as np
import math
from matplotlib import pyplot as plt
import os, glob, json

machine_epsilon = np.finfo(float).eps

class ReferenceImage:
    def __init__(self, image_data):
        """
        This class is designed around 3 possible case:
        3D reference, 3D target (e.g., FF-OCT volumes)
        3D reference, 2D target (e.g., AO-OCT b-scan registration)
        2D reference, 2D target (e.g., serial B-scanning 2D registration)
        """
        self.image_data = image_data

        self.fref = np.conj(np.fft.fftn(self.image_data))

        try:
            assert self.fref.ndim in [2,3]
        except AssertionError as ae:
            sys.exit('%s. Reference image has dimensionality %d but should be 2 or 3.'%(ae,self.fref.ndim))

    def __str__(self):
        return 'ReferenceImage object holding data with shape %s and dtype %s.'%(self.image_data.shape,self.image_data.dtype)
     
    @staticmethod
    def _wrap_fix(p, size):
        # identical wrap convention used in testing_broadcasting.py (fix on z,x; not y)
        return p if p < size // 2 else p - size
    
    def register(self, target_image, poxc=True):
        """
        The reference image has shape [slow, depth, fast], [depth, fast].

        This function registers a target image to the reference image.

        The target may have the same dimensionality as the reference image or it
        may have lower dimensionality. If it has the same dimensionality, it should
        have the same shape.

        If it has lower dimensionality, its trailing dimensions must match those of the
        reference data. E.g., reference data with shape (S,D,F) can accommodate registration
        with targets of shapes (S,D,F), (D,F) or (F). If the target has lower dimensionality
        its n-dim DFT is multiplied by that of the reference image using broadcasting.

        """

        # originally we take the conjugate of the target, but why not do this just
        # once for the reference, and then change the sign of the resulting delays?
        #ftar = np.conj(np.fft.fftn(target_image))
        ftar = np.fft.fftn(target_image)
        
        # broadcast multiply across slow dimension:
        prod = self.fref * ftar                              # (slow, depth, fast)
        if poxc:
            prod = prod / (np.abs(prod) + machine_epsilon)
            
        # NOTE: testing_broadcasting computed ifftn(self.fref*ftar) again.
        # We keep the same behavior outcome by using 'prod' here.
        xc_arr = np.abs(np.fft.ifftn(prod))                  # (slow, depth, fast)


        # Previous method was to specify semantics for the dimensions of the output
        # and output a 4-tuple. This is a little bit rigid and won't work in unexpected
        # cases, such as a 2D reference.
        
        # # Peak & wrap the same way the script does (wrap z,x only)
        # yp, zp, xp = np.unravel_index(np.argmax(xc_arr), xc_arr.shape)
        # sy, sz, sx = xc_arr.shape
        # zp = self._wrap_fix(zp, sz)
        # xp = self._wrap_fix(xp, sx)
        
        # xc_max = float(np.max(xc_arr))
        # # return dict(dx=xp, dy=yp, dz=zp, xc=float(np.max(xc_arr)))
        # return yp, zp, xp, xc_max
    
        # Given our need for flexible dimensions, let's be agnostic about their orientations
        # and just spit them out in the right order, in a dictionary

        shifts_tuple = np.unravel_index(np.argmax(xc_arr), xc_arr.shape)

        d1 = self._wrap_fix(shifts_tuple[1],xc_arr.shape[1])
        d2 = self._wrap_fix(shifts_tuple[2],xc_arr.shape[2])
        if ftar.ndim==3 and self.fref.ndim==3:
            d0 = self._wrap_fix(shifts_tuple[0],xc_arr.shape[0])
        elif ftar.ndim==2 and self.fref.ndim==3:
            d0 = shifts_tuple[0]
        elif ftar.ndim==2 and self.fref.ndim==2:
            d0 = None

        xc_max = np.max(xc_arr)

        result = {'d0':d0,'d1':d1,'d2':d2,'xc_max':xc_max}
        return result
            
            


            
