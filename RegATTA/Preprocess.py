# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 12:51:13 2025

@author: ZAQ
"""

import numpy as np

class Preprocess:
    def __init__(self, folder):
            self.folder = folder
            
    @staticmethod
    def _rfunc():
        import importlib
        try:
            # Prefer relative import when running as a package
            return importlib.import_module(".RegistrationFunctions", __package__)
        except Exception:
            # Fallback: absolute import if package context is missing
            return importlib.import_module("RegATTA.RegistrationFunctions")

    @classmethod       
    # ---- loaders / cropping ----
    def load_volume_from_folder(cls, folder: str, prefix: str = "bscan", *, crop: bool = False):
        """
        Wrapper over rfunc.get_volume / get_volume_and_crop (returns shape (slow, depth, fast)).
        """
        rfunc = cls._rfunc()
        vol = rfunc.get_volume(folder, prefix=prefix)            # (slow, depth, fast)
        return vol
    
    # def auto_crop_volume(cls, volume: np.ndarray):
    #     """
    #     Ask rfunc.auto_crop for z1/z2 based on z-profile, then slice.
    #     Expects (slow, depth, fast); returns cropped volume with same ordering.
    #     """
    #     rfunc = cls._rfunc()
    #     z1, z2 = rfunc.auto_crop(volume)
    #     return volume[:, z1:z2, :], (int(z1), int(z2))
    
    @classmethod
    def load_crop_volume(cls, folder):
        """
        Ask rfunc.auto_crop for z1/z2 based on z-profile, then slice.
        Expects (slow, depth, fast); returns cropped volume with same ordering.
        """
        rfunc = cls._rfunc()
        volume = cls.load_volume_from_folder(folder)
        z1, z2 = rfunc.auto_crop(volume)
        return volume[:, z1:z2, :]