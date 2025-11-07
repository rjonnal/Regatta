# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 11:42:26 2025

@author: ZAQ
"""
from .Preprocess import Preprocess
import numpy as np
import math
from matplotlib import pyplot as plt
import os, glob, json

class ReferenceImage:
    def __init__(self, vol_dir):
            self.vol = Preprocess.load_crop_volume(vol_dir)
            self.fref = np.fft.fftn(self.vol)
            self.n_slow, self.n_depth, self.n_fast = self.vol.shape
            # self.sy, self.sz, self.sx = vol.shape
    
    @staticmethod
    def _wrap_fix(p, size):
        # identical wrap convention used in testing_broadcasting.py (fix on z,x; not y)
        return p if p < size // 2 else p - size
    
    def register(self, target_bscan, poxc=True):
        """
        Register a single B-scan (shape: [depth, fast]) to the 3D reference.
        Uses broadcasting: FFT2(target) conj, multiply into fref, IFFTN, argmax.
        poxc=True => normalized cross-power (phase correlation) like original.
        """
        ftar = np.conj(np.fft.fft2(target_bscan))            # (depth, fast)
        # broadcast multiply across slow dimension:
        prod = self.fref * ftar                              # (slow, depth, fast)
        if poxc:
            prod = prod / (np.abs(prod) + 1e-12)
        # NOTE: testing_broadcasting computed ifftn(self.fref*ftar) again.
        # We keep the same behavior outcome by using 'prod' here.
        xc_arr = np.abs(np.fft.ifftn(self.fref * ftar))                  # (slow, depth, fast)

        # Peak & wrap the same way the script does (wrap z,x only)
        yp, zp, xp = np.unravel_index(np.argmax(xc_arr), xc_arr.shape)
        sy, sz, sx = xc_arr.shape
        zp = self._wrap_fix(zp, sz)
        xp = self._wrap_fix(xp, sx)
        
        xc_max = float(np.max(xc_arr))
        # return dict(dx=xp, dy=yp, dz=zp, xc=float(np.max(xc_arr)))
        return yp, zp, xp, xc_max
    