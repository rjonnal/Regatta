# RegATTA:  Registration for Advanced Three-dimensional Tomographic Analysis
_RegATTA_ is a ... (general description)
The _RegATTA_ project is supported by (funding number)

## Design Doc (For developing stage usage, tmp)
### What do we want this model to do?
- interactively display reference image and target image. Synchronize

### Workflow of registering real data
- Load oct volumes.
  - Convert any input format to structure /Subject/Subject_bscans/VolumeIndex/XXX.npy
- Define reference volume index, display 1 slice of bscan and en face projection.
  - Should have an option of "auto defined ref" or "user defined ref". If "auto defined ref", should display the tests and metrics that determine the current selection of the reference volume.
- Simulation for training purpose
- preprocessing
   - Crop
   - Flatten
   - Upsampling
- Register
   - Broadcasting
   - 3D strip based
   - coarse-to-fine
- Correct Volumes
- Visual comparison between ref image, and selected post-reg image

## Table of Contents
