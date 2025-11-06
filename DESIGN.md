# Design document for OCT registration software

## General description

This software will permit users to align sequential OCT images for the purposes of averaging images to reduce noise or tracking image features to monitor physiological or other changes.

## Requirements (indicate priority)

1. (high) Given input of a series of OCT volumes, produce as output a series of transformed volumes such that:
   a. A set of 3D coordinates will select the same object feature in each of the volumes in which that feature was imaged.
   b. Averaging the transformed volumes will produce an improved image.

2. (low) Produce a 3D trace of the object's movement during imaging.

3. (high) Produce diagnostic plots/logs etc. at each step in the processing to help with debugging/troubleshooting.

4. (medium) Generate synthetic data for testing.

5. (high) Provide tools for qualitative evaluation of registration performace. I.e., apples to apples comparisons between single volumes and registered averages.

6. (high) Provide figures of merit to objectively assess the performance. Could use Pearson correlation, Dice coefficient, or RMS error.

7. (high) Software should work on arbitrarily long series of volumetric images.

8. (high) Should work on serial B-scans or serial A-scans.

9. (high) Should permit global registration (all images to a single reference) or sequential registration (each image in a series registered to the preceding image) with possibility of integration.

10. (high) Should work on rigid volumes, e.g., FF-OCT data.

## Use cases

```
import regatta as rg

reference_folder = 'path/to/reference/bscans'

refim = rg.ReferenceImage(reference_folder)

def load_image(folder):
    return np.ndarray([np.load(f) for f in sorted(glob.glob(os.path.join(folder,'*.npy')))])
    
ris = rg.RegisteredImageSeries(load_function=load_image)

# target data exist in numbered subfolders of mydata:
target_folders = sorted(glob.glob('mydata/00*'))

for tf in target_folders:
    ris.add(tf)
    # ris knows how to add an image from a folder, using custom or default load_image function
    # when adding the image, ris doesn't load the data, it just stores the value of tf and
    # initializes dy, dz, dx, xc maps to zero
    # how does ris know dimensions of data? maybe initialize w/ dimensions, or load one test image
    target = ris.get(tf)

    # for rigid bodies (including serial B-scans)
    dy,dz,dx,xc = refim.register(target,phase_only=False)
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

```

## Design (classes, functions, constants, configuration files, manifests)

### General ideas

1. Overall design will be hybrid object-oriented/functional.

2. Don't worry too much about speed/optimization at this stage, but keep the following in mind: DFT acceleration using CuPy or numba; the data will be too big to keep in RAM at all once, but we also want to minimize disk access.

### Module organization

For now, let's put each class in its own module; modules should named registered_image_series, and classes should be named in camel case like RegisteredImageSeries. Functions go in another module called functions.py.

### Classes

#### `ReferenceImage`

Purpose: represents the reference image and stores its own N-dim DFT to avoid repeated DFTs for every target image

Data:

	* image data, numpy ndarray
	
	* FFTN of image data, numpy ndarray

Functions:

	* register: input a target image (either B-scan, volume chunk, or full volume) and return coordinates of its location in the reference image, the peak of the cross-correlation used to register it, and other measures of correlation (Dice coefficient, Pearson correlation) applied to the amplitude and the phase


#### `RegisteredImageSeries`

Purpose: store registration coordinates and locations (folders or filenames) containing image data

Data:

	* folders/filenames, one per image in series

	* 2D (slow x fast dimensions) maps of: dy, dz, dx, correlation (cross-correlation peak), maybe other figures of merit; these are 2D because we treat A-scans as rigid bodies

Functions:

	* finalize: tells RIS that all data is registered; internally should calculate shape of
	  expanded volumes

	* get: create an expanded volume of nans and load and put the image data into the
	  correct locations in the expanded volume using the maps

### Documentation

Use [Google-style Python docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html), but don't go overboard! We'll later use [lazydocs](https://github.com/ml-tooling/lazydocs) to create github/gitlab Markdown for a simple API reference page.



