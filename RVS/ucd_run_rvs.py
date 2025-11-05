# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 13:50:49 2025

@author: AQZ
Updated 10/27/2025
"""

from ucd_registered_volume_series import RegisteredVolumeSeries as RVS
from datetime import datetime
import os

#%% ============ Synthetic Data ==================

#%% 1) Simulate a small dataset (writes bscans and traces to disk)
RVS.simulate_synthetic_data(
    output_root='bscans_synthetic',
    info_dir='bscans_synthetic_info',
    n_volumes=2
)

#%% 2) Register synthetic vol 1 → vol 0 and compare against traces
ysh, zsh, xsh = RVS.register_synthetic_pair(
    data_root='bscans_synthetic',
    ref_index=0,
    target_index=1,
    sigma=5.0,
    compare_to_trace=True,
    info_folder='bscans_synthetic_info',
    show_plots=True
)

#%% 3) Broadcasting registration on volumes loaded from folders 00000..00005

# base folder
base_dir = r"Fast"
folder_prefix = "boradcasting_"

# Add timestamp (yearmonthday_hourminute)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Combine
save_dir = os.path.join(base_dir, f"{folder_prefix}_{timestamp}")

# Make sure it exists
os.makedirs(save_dir, exist_ok=True)

print(save_dir)

res = RVS.register_by_broadcasting_from_root(
    # root=r'F:\Registration\Fast\locNS05_b6_bscans',
    root=r'Fast/locNS05_b6_bscans',
    indices=[0,1,2,3,4,5],
    poxc=True,
    plot=True,
    save_dir=r'Fast/boradcasting_10232025'
)
corrected = res["corrected_volumes"]

#%% 4) End-to-end helper (load → optional upsample/flatten → register → correct)
out = RVS.build_and_register_from_root(
    # root=r'F:\Registration\Fast\locNS05_b6_bscans',
    root=r'Fast/locNS05_b6_bscans',
    prefix='bscan',
    indices=[0,1,2,3,4,5],
    upsample_factor=None,   # or 2
    flatten_x=False,        # placeholder (identity) flattener; swap if you want
    use_broadcasting=True,
    plot=True
)
