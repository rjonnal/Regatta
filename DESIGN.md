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

1. Overall design will use classes when useful and functions when useful.

2.

### Classes

`ReferenceVolume`: 

