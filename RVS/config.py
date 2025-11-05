import glob
import os
import numpy as np
# data (.bin) file format parameters
initial_skip = 491 #491 for 800/sweep, 658 for 1152/sweep
samples_per_ascan = 2304 #2304
n_volumes = 5 #20#42#20
bscans_per_volume = 172 #172  #200
ascans_per_bscan = 160# 160 #414
# n_volumes = 21weqa
# bscans_per_volume = 1
# ascans_per_bscan = 414
bytes_per_sample = 2
extra_samples_per_bscan = 1177 #2436
dtype = '<i2'

GlobalRef = False
refVidx = 3

deltaT = 0.035 #0.035 #35ms/volume
Nbaseline = 2
# bscan cropping parameters
zeropad_fold = 4
bscan_z1 = 60
# bscan_z1 = 250  #80*zeropad_fold
# bscan_z2 = bscan_z1 + 150*zeropad_fold #bscan_z1+150*zeropad_fold
# bscan_z1 = 0#2150
bscan_z2 = 400


# processing parameters
# mapping_index_filename = r'C:\Users\rjonn\OneDrive\0.CHOIR_YC\4.Code\FDML_processing_MWE_v2\FDML_org\mapping_indices_2304.ind'
# mapping_index_filename = r'D:\OneDrive\0.CHOIR_YC\4.Code\FDML_processing_MWE_v2\FDML_org\mapping_indices_2304.ind'
file1 = glob.glob("*kmap_Z800.txt")  #*mapping_indices_2304.ind
if file1:
    mapping_index_filename = os.path.abspath(file1[0])  # Get full path of the first file
    # print(mapping_index_filename)
else:
    print("No .ind file found.")

# laser parameters
L0 = 1058e-9
delta_L = 76e-9 #78
refractiveindex = 1.38
# dispersion compensation parameters
do_dispersion_compensation = True
dispersion_volume_index = 1
dispersion_bscan_index = [10]
dispersion_dB_clims = (80,110)
dc3_abs_max = 1e-17
dc2_abs_max = 1e-10



# scanning parameters
resonant_scanner_frequency = 2000.0
resonant_scanner_amplitude_deg = 1.0

# dewarping parameters
dewarping_volume_index = 1 #0
dewarping_bscan_index = 10
default_scan_range_radians =  3.36 # 3.31
default_scan_offset_radians = -0.025 #0 # radians
scan_angle_radians_abs_max = 6.3
scan_offset_radians_abs_max = 3.14159

# sampling parameters
sampling_interval_x_m = 300e-6/ascans_per_bscan #0.72e-6
sampling_interval_y_m = 300e-6/bscans_per_volume #1.5e-6
sampling_interval_z_um = 7.36/zeropad_fold  #7.69 um/pixel in air without zero padding

# multiprocessing
multiprocessing = False
n_processors = 10

# automated z cropping
desired_depth_px = 350 #100*zeropad_fold

# projection and cone segmentation settings
# number of pixels to pad axially when segmenting and projecting photoreceptors
z_padding_px = 5*zeropad_fold #5


projections_dpi = 300

# segmentation UI settings
layer_markers = ['ro','gs','bd','c^']
layer_markersize = 4
layer_markersize_print = 1
layer_markeralpha = 0.4

# strip registration
# default strip width parameter
default_sigma = 5 #5

# maximum (aboslute) displacement of a strip relative to reference frame
max_displacement = 150 #50
minimum_correlation = 0.01 #0.012

# power spectrum (dB) contrast limits
ps_clim = (240,280)

# cell localization and segmentation
# the lateral dimensions of the segmented cell will be 2*cell_radius+1--always odd
#fast sampling 0.725um; slow sampling 1.5 um; cell size around 8.8 um at 6T
cell_half_width_fast = 3 #5
cell_half_width_slow = 3
Lcone = 4*zeropad_fold # the OS length is around 23um in tissue = 31.74um in air
# the displacement of ISOS and cost layers at 6T is thus around 31.74/(7.69/zeropad_fold)=16 pixels in depth


projection_smoothing_sigma = 1 # 1.5 for cone identification

# refinement of cone localization
# after strip registration, the cones are localized again in the reference
# frame using local cross correlation; a few parameters are required:
refinement_x_half_width=15
refinement_y_half_width=3
# refinement shifts greater than this value are presumed to be noise and ignored
refinement_max_displacement=3

#threshold for identify good cells
depth_correlation = 0.5

