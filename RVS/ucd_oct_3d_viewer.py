# -*- coding: utf-8 -*-

"""
Created on Thu Sep 11 21:51:16 2025

OCT 3D Volume Visualizer (slices + MIPs + optional 3D isosurface)

Features:
- Fast orthogonal slice viewer with keyboard and slider controls.
- Three-way MIP (max intensity projection).
- Optional 3D isosurface rendering (if scikit-image + plotly are installed).
- Flexible axis order to match OCT conventions.
  - Many pipelines use (slow, depth, fast) or (depth, slow, fast). You can select with --axes.

Usage:
    python oct_3d_viewer.py --input path/to/volume.npy --axes slow depth fast
    python oct_3d_viewer.py --input path/to/volume.npy --axes depth slow fast
    python oct_3d_viewer.py --folder path/to/slices --pattern "*.tif" --axes depth slow fast
    python oct_3d_viewer.py --isosurface 0.7

Keyboard shortcuts in the viewer:
    j/k : previous/next slice in the primary axis (default: depth)
    h/l : previous/next slice in the second axis
    u/i : previous/next slice in the third axis
    a/z : decrease/increase contrast lower bound (vmin)
    s/x : decrease/increase contrast upper bound (vmax)
    r   : reset window/level to data percentiles
    m   : toggle MIP grid view
    g   : toggle Guide
    q   : quit

Sliders on the right control the three slice indices.

@author: AQZ
Updated: 09112025
"""
import argparse
import glob
import os
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

try:
    import imageio.v3 as iio  # modern imageio
except Exception:
    iio = None

# Optional dependencies (for isosurface 3D)
try:
    from skimage.measure import marching_cubes
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False


AXES_CHOICES = ["depth", "slow", "fast"]


def load_npy(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D array in {path}, got shape {arr.shape}")
    return arr


def load_folder(folder: str, pattern: str) -> np.ndarray:
    if iio is None:
        raise RuntimeError("imageio.v3 not available. Install 'imageio' to read image stacks.")
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        raise FileNotFoundError(f"No images found in {folder!r} with pattern {pattern!r}")
    stack = [iio.imread(p) for p in paths]
    vol = np.stack(stack, axis=0)  # (num_slices, H, W)
    if vol.ndim != 3:
        raise ValueError(f"Loaded stack is not 3D. Got shape {vol.shape}")
    return vol


def reorder_axes(vol: np.ndarray, axes_order: Tuple[str, str, str]) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Reorder the input volume to internal order (depth, slow, fast) = (Z, Y, X).
    axes_order describes current order of vol: e.g., ('slow','depth','fast') etc.
    Returns (vol_z_y_x, original_order_indices)
    """
    mapping = {name: i for i, name in enumerate(axes_order)}
    try:
        z_axis = mapping["depth"]
        y_axis = mapping["slow"]
        x_axis = mapping["fast"]
    except KeyError as e:
        raise ValueError(f"axes_order must contain exactly these three labels: {AXES_CHOICES}. Got {axes_order}") from e

    perm = (z_axis, y_axis, x_axis)
    vol_reordered = np.transpose(vol, perm)
    return vol_reordered, perm


def percentile_window(vol: np.ndarray, p_lo=1.0, p_hi=99.0) -> Tuple[float, float]:
    vmin, vmax = np.percentile(vol[np.isfinite(vol)], (p_lo, p_hi))
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


class SliceViewer3D:
    def __init__(self, vol_zyx: np.ndarray, fname):
        if vol_zyx.ndim != 3:
            raise ValueError("SliceViewer3D expects a 3D array (Z,Y,X).")
        self.vol = vol_zyx
        self.Z, self.Y, self.X = vol_zyx.shape
        self.idx_z = self.Z // 2
        self.idx_y = self.Y // 2
        self.idx_x = self.X // 2
        self.show_mip = False
        self.show_guides = True   # show/hide overlays
        self.fname = fname
	
        self.vmin, self.vmax = percentile_window(self.vol, 1, 99)

        self._build_ui()
        self._connect_events()
        self._redraw()

    def _build_ui(self):
        plt.close('all')
        self.fig = plt.figure(figsize=(12, 6))

        # Main axes (left: three orthogonal slices in a 2x2 grid)
        self.ax_xy = self.fig.add_axes([0.05, 0.55, 0.4, 0.4])  # XY at current Z (depth slice)
        self.ax_xz = self.fig.add_axes([0.55, 0.55, 0.4, 0.4])  # XZ at current Y
        self.ax_yz = self.fig.add_axes([0.05, 0.05, 0.4, 0.4])  # YZ at current X

        # Sliders on the right side
        slider_left = 0.55
        slider_width = 0.4
        self.ax_s_z = self.fig.add_axes([slider_left, 0.45, slider_width, 0.03])
        self.ax_s_y = self.fig.add_axes([slider_left, 0.40, slider_width, 0.03])
        self.ax_s_x = self.fig.add_axes([slider_left, 0.35, slider_width, 0.03])

        self.s_z = Slider(self.ax_s_z, "Depth (Z)", 0, self.Z-1, valinit=self.idx_z, valstep=1)
        self.s_y = Slider(self.ax_s_y, "Slow (Y)",  0, self.Y-1, valinit=self.idx_y, valstep=1)
        self.s_x = Slider(self.ax_s_x, "Fast (X)",  0, self.X-1, valinit=self.idx_x, valstep=1)

        # Contrast sliders
        self.ax_s_vmin = self.fig.add_axes([slider_left, 0.25, slider_width, 0.03])
        self.ax_s_vmax = self.fig.add_axes([slider_left, 0.20, slider_width, 0.03])
        self.s_vmin = Slider(self.ax_s_vmin, "vmin", 0.0, 1.0, valinit=0.0, valstep=0.001)
        self.s_vmax = Slider(self.ax_s_vmax, "vmax", 0.0, 1.0, valinit=1.0, valstep=0.001)

        # Buttons
        self.ax_btn_reset = self.fig.add_axes([slider_left, 0.12, 0.1, 0.05])
        self.ax_btn_mip   = self.fig.add_axes([slider_left+0.14, 0.12, 0.1, 0.05])
        self.ax_btn_guides = self.fig.add_axes([slider_left+0.28, 0.12, 0.1, 0.05])
        self.btn_reset = Button(self.ax_btn_reset, "Reset WL")
        self.btn_mip   = Button(self.ax_btn_mip,   "Toggle MIP")
        self.btn_guides = Button(self.ax_btn_guides, "Toggle Guides")

        # Status text
        self.ax_txt = self.fig.add_axes([slider_left, 0.02, slider_width, 0.08])
        self.ax_txt.axis('off')
        self.text = self.ax_txt.text(0.0, 0.5, "", fontsize=10, va='center', ha='left')

        self.fig.suptitle(f"OCT 3D Visualizer — {self.fname}", y=0.98)

    def _connect_events(self):
        self.s_z.on_changed(lambda v: self._set_idx('z', int(v)))
        self.s_y.on_changed(lambda v: self._set_idx('y', int(v)))
        self.s_x.on_changed(lambda v: self._set_idx('x', int(v)))
        self.s_vmin.on_changed(lambda v: self._set_window())
        self.s_vmax.on_changed(lambda v: self._set_window())
        self.btn_reset.on_clicked(lambda evt: self._reset_window())
        self.btn_mip.on_clicked(lambda evt: self._toggle_mip())
        self.btn_guides.on_clicked(lambda evt: self._toggle_guides())
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
    def _toggle_guides(self):
        self.show_guides = not self.show_guides
        self._redraw()
    
    def _overlay_guides_xy(self):
        # XZ plane location (Y index) → horizontal line on XY
        self.ax_xy.axhline(self.idx_y, linewidth=1)
        # YZ plane location (X index) → vertical line on XY
        self.ax_xy.axvline(self.idx_x, linewidth=1)
        # Labels (optional)
        # self.ax_xy.text(2, self.idx_y+2, f"XZ @ Y={self.idx_y}", fontsize=9,
        #                 va="bottom", ha="left", bbox=dict(fc="w", alpha=0.5, lw=0))
        # self.ax_xy.text(self.idx_x+2, 2, f"YZ @ X={self.idx_x}", fontsize=9,
        #                 va="top", ha="left", rotation=90, bbox=dict(fc="w", alpha=0.5, lw=0))
    
    def _overlay_guides_xz(self):
        # XZ image is shape (Z, X): Z is vertical axis → mark current Z with a horizontal line
        self.ax_xz.axhline(self.idx_z, linewidth=1)
        # self.ax_xz.text(2, self.idx_z+2, f"Z={self.idx_z}", fontsize=9,
        #                 va="bottom", ha="left", bbox=dict(fc="w", alpha=0.5, lw=0))
    
    def _overlay_guides_yz(self):
        # YZ image we draw is (Y,Z)=vol[:, :, X].T → Z is horizontal axis → vertical line at current Z
        self.ax_yz.axvline(self.idx_z, linewidth=1)
        # self.ax_yz.text(self.idx_z+2, 2, f"Z={self.idx_z}", fontsize=9,
        #                 va="top", ha="left", rotation=90, bbox=dict(fc="w", alpha=0.5, lw=0))

        
    def _set_idx(self, axis, idx):
        if axis == 'z':
            self.idx_z = int(np.clip(idx, 0, self.Z-1))
        elif axis == 'y':
            self.idx_y = int(np.clip(idx, 0, self.Y-1))
        elif axis == 'x':
            self.idx_x = int(np.clip(idx, 0, self.X-1))
        self._redraw()

    def _set_window(self):
        # map sliders [0,1] to [vmin,vmax] span around percentile_window
        pmin, pmax = percentile_window(self.vol, 1, 99)
        vmin = pmin + (pmax - pmin) * self.s_vmin.val
        vmax = pmin + (pmax - pmin) * self.s_vmax.val
        if vmax <= vmin:
            vmax = vmin + 1e-6
        self.vmin, self.vmax = float(vmin), float(vmax)
        self._redraw(refresh_text=False)

    def _reset_window(self):
        self.vmin, self.vmax = percentile_window(self.vol, 1, 99)
        self.s_vmin.set_val(0.0)
        self.s_vmax.set_val(1.0)
        self._redraw()

    def _toggle_mip(self):
        self.show_mip = not self.show_mip
        self._redraw()

    def _on_key(self, event):
        key = event.key
        if key in ('q', 'escape'):
            plt.close(self.fig)
            return
        # navigation
        if key == 'j': self._set_idx('z', self.idx_z-1)
        if key == 'k': self._set_idx('z', self.idx_z+1)
        if key == 'h': self._set_idx('y', self.idx_y-1)
        if key == 'l': self._set_idx('y', self.idx_y+1)
        if key == 'u': self._set_idx('x', self.idx_x-1)
        if key == 'i': self._set_idx('x', self.idx_x+1)
        # window/level
        if key == 'a': self.vmin -= 0.02*(self.vmax-self.vmin); self._redraw()
        if key == 'z': self.vmin += 0.02*(self.vmax-self.vmin); self._redraw()
        if key == 's': self.vmax -= 0.02*(self.vmax-self.vmin); self._redraw()
        if key == 'x': self.vmax += 0.02*(self.vmax-self.vmin); self._redraw()
        if key == 'r': self._reset_window()
        if key == 'm': self._toggle_mip()
        if key == 'g': self._toggle_guides()

    def _draw_slice(self, ax, img, title):
        ax.clear()
        ax.imshow(img, cmap='gray', vmin=self.vmin, vmax=self.vmax, origin='upper', aspect='auto')
        # ax.imshow(img, cmap='gray', vmin=self.vmin, vmax=self.vmax, aspect='auto')
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    def _draw_mip_grid(self):
        # Three MIPs: Z (along depth), Y, X
        mip_z = np.max(self.vol, axis=0)  # (Y,X)
        mip_y = np.max(self.vol, axis=1)  # (Z,X)
        mip_x = np.max(self.vol, axis=2)  # (Z,Y)

        self._draw_slice(self.ax_xy, mip_z, "MIP (over Z) → YX")
        self._draw_slice(self.ax_xz, mip_y, "MIP (over Y) → ZX")
        self._draw_slice(self.ax_yz, mip_x.T, "MIP (over X) → YZ")  # transpose to match orientation

    def _redraw(self, refresh_text=True):
        if self.show_mip:
            self._draw_mip_grid()
        else:
            # xy = self.vol[self.idx_z, :, :]        # Z fixed → image in Y-X
            xy = self.vol[self.idx_z, :, :]        # Z fixed → image in Y-X
            xz = self.vol[:, self.idx_y, :]        # Y fixed → image in Z-X
            yz = self.vol[:, :, self.idx_x].T      # X fixed → image in Y-Z (transpose for display)

            self._draw_slice(self.ax_xy, xy, f"Slice XY @ Z={self.idx_z}")
            if self.show_guides:
                self._overlay_guides_xy()
            
            self._draw_slice(self.ax_xz, xz, f"Slice XZ @ Y={self.idx_y}")
            if self.show_guides:
                self._overlay_guides_xz()
            
            self._draw_slice(self.ax_yz, yz, f"Slice YZ @ X={self.idx_x}")
            if self.show_guides:
                self._overlay_guides_yz()


        if refresh_text:
            self.text.set_text(
                f"Shape (Z,Y,X) = {self.vol.shape}\n"
                f"indices: Z={self.idx_z}, Y={self.idx_y}, X={self.idx_x}\n"
                f"window/level: vmin={self.vmin:.3g}, vmax={self.vmax:.3g}"
            )
        self.fig.canvas.draw_idle()

    # -------- Optional: 3D isosurface with plotly (nice but slower) --------
    def show_isosurface(self, level: Optional[float] = None):
        if not (_HAS_SKIMAGE and _HAS_PLOTLY):
            print("Isosurface requires scikit-image and plotly. Try: pip install scikit-image plotly")
            return
        vol = self.vol.astype(np.float32)
        vmin, vmax = percentile_window(vol, 1, 99)
        if level is None:
            level = 0.5 * (vmin + vmax)
        verts, faces, _, _ = marching_cubes(vol, level=level, spacing=(1.0, 1.0, 1.0))
        x, y, z = verts[:, 2], verts[:, 1], verts[:, 0]  # map back to X,Y,Z
        mesh = go.Mesh3d(x=x, y=y, z=z, i=faces[:,0], j=faces[:,1], k=faces[:,2], opacity=1.0, flatshading=True)
        fig = go.Figure(data=[mesh])
        fig.update_layout(title=f"Isosurface @ level={level:.3g}", scene=dict(
            xaxis_title="X (fast)", yaxis_title="Y (slow)", zaxis_title="Z (depth)"
        ))
        fig.show()


def main():
    parser = argparse.ArgumentParser(description="OCT 3D Volume Visualizer")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=str, help="Path to a .npy 3D array")
    src.add_argument("--folder", type=str, help="Folder containing 2D image slices")
    parser.add_argument("--pattern", type=str, default="*.png", help="Glob pattern for --folder (e.g., '*.tif')")
    parser.add_argument("--axes", nargs=3, metavar=("A0","A1","A2"), default=["depth","slow","fast"],
                        help="Axis names (in order) of the loaded data. Each must be one of: depth slow fast. "
                             "They will be mapped to internal (depth, slow, fast).")
    parser.add_argument("--isosurface", type=float, default=None,
                        help="If set, also open a 3D isosurface window at this intensity level (percentile if 0<val<=1, else raw value).")
    args = parser.parse_args()

    # Load data
    if args.input:
        vol = load_npy(args.input)
    else:
        vol = load_folder(args.folder, args.pattern)
    
    fname = os.path.splitext(os.path.basename(args.input))[0]
    
    # Ensure values are positive (OCT intensities usually are)
    vol = np.abs(vol)
    
    # Apply log transform
    vol_log = np.log1p(vol)   # log(1 + vol), avoids log(0)
    
    # Normalize to [0,1] for display
    vol_log = (vol_log - vol_log.min()) / (vol_log.max() - vol_log.min())
    
    # Map axes to (Z,Y,X) = (depth, slow, fast)
    axes_order = tuple(args.axes)
    for a in axes_order:
        if a not in AXES_CHOICES:
            raise ValueError(f"Invalid axis label {a!r}. Choose from {AXES_CHOICES}")
    vol_zyx, _ = reorder_axes(vol, axes_order)

    # Normalize to [0,1] for display if not already
    vmin, vmax = percentile_window(vol_zyx, 1, 99)
    vol_disp = np.clip((vol_zyx - vmin) / (vmax - vmin + 1e-12), 0, 1).astype(np.float32)

    viewer = SliceViewer3D(vol_disp, fname)
    plt.show()

    # Optional isosurface
    if args.isosurface is not None:
        level = args.isosurface
        if 0 < level <= 1.0:
            # treat as percentile fraction between [vmin, vmax]
            level = vmin + level * (vmax - vmin)
        viewer.show_isosurface(level)


if __name__ == "__main__":
    main()
