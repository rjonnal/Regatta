# RegATTA:  Registration for Advanced Three-dimensional Tomographic Analysis
_RegATTA_ is a unified framework for managing, preprocessing, registering, and visualizing OCT volume data. It supports flexible data ingestion by converting any input format into a standardized directory structure. Users can define a reference volume manually or enable an auto-selection mode, which presents diagnostic metrics and visual tests to justify the chosen reference. The pipeline includes optional simulation modules for training, as well as preprocessing tools such as cropping, flattening, and upsampling. Multiple registration strategies are available—including broadcasting-based alignment, 3D strip-based registration, and coarse-to-fine iterative correction—to accommodate different motion profiles and imaging scenarios. After registration, the _RegATTA_ can generate corrected volumes and provides visual comparison tools to evaluate differences between the reference image and any post-registration volume, enabling transparent validation of registration performance.
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
- Register Options
   - Broadcasting
   - 3D strip based
   - coarse-to-fine
- Correct Volumes
- Possible segmentation and cone identification
- Visual comparison between ref image, and selected post-reg image

## Table of Contents
