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

## Design (classes, functions, constants, configuration files, manifests)

### General ideas

1. Overall design will be hybrid object-oriented/functional.

2. Don't worry too much about speed/optimization at this stage, but keep the following in mind: DFT acceleration using CuPy or numba; the data will be too big to keep in RAM at all once, but we also want to minimize disk access.

### Classes

#### `ReferenceVolume`

Justification: it can store its own N-dim DFT to avoid repeated DFTs for every target image

Data:
	* image data, numpy ndarray
	* FFTN of image data, numpy ndarray

Functions:
	* register: input a target image (either B-scan, volume chunk, or full volume) and return coordinates of its location in the reference image, the peak of the cross-correlation used to register it, and other measures of correlation (Dice coefficient, Pearson correlation) applied to the amplitude and the phase


### Documentation

Use [Google-style Python docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html), but don't go overboard! We'll later use [lazydocs](https://github.com/ml-tooling/lazydocs) to create github/gitlab Markdown for a simple API reference page.



