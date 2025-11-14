"""
Created on Thu Nov 13
@author: RSJ

In this test, we use an image-loading function from regatta.io, and use the
resulting data to instantiate a ReferenceImage object.
"""
from regatta.io import load_image_aooct as load_image
from regatta.reference_image import ReferenceImage
import numpy as np

location = '/home/rjonnal/projects/volume_registration/bscans_aooct/00000'

reference_data = load_image(location)

ref = ReferenceImage(reference_data)

