"""
Created on Thu Nov 13
@author: RSJ

Because RegATTA is designed to work with various kinds of OCT data
and file types, the user/caller is expected to provide a function that
loads images. It's signature should be:

```
def load_image(location):
    ...
    ...
    return image
```

`location` can be a folder or a file containing an image.
`image` should be a numpy array

Pre-written load functions are included in regatta.io for AO-OCT and clinical ORG data.
"""


location = '/home/rjonnal/projects/volume_registration/bscans_aooct/00000'

# Test 1:

import glob,os
import numpy as np
def load_image(location):
    files = glob.glob(os.path.join(location,'bscan*.npy'))
    files.sort()
    image = [np.load(f) for f in files]
    dtype = image[0].dtype
    image = np.array(image,dtype=dtype)
    return image

vol = load_image(location)
print(vol.shape)

# Test 2:

from regatta.io import load_image_aooct as load_image

vol = load_image(location)
print(vol.shape)
