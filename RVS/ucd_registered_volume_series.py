import numpy as np
import math
from matplotlib import pyplot as plt
import os, glob, json


#%% RVS Class
class RegisteredVolumeSeries:
    
    #%%
    def __init__(self):
        self.volumes = []
        self.xshifts = []
        self.yshifts = []
        self.zshifts = []


    def add(self,volume,x=None,y=None,z=None):

        if x is None or y is None or z is None:
            sy,sz,sx = volume.shape
            x = np.zeros((sy,sx))
            y = np.zeros((sy,sx))
            z = np.zeros((sy,sx))
        
        self.volumes.append(volume)
        self.xshifts.append(x)
        self.yshifts.append(y)
        self.zshifts.append(z)
    
    
    # def correct_volumes(self):
    #     """
    #     Build corrected volumes on a common canvas.
    #     Assumes each volume has shape (sy, sz, sx) = (Y, Z, X).
    #     Shift maps are per-(y,x) arrays (same sy×sx) giving subpixel shift in voxels.
    #       - xshifts, yshifts: applied to X/Y as *rounded* integer pixels.
    #       - zshifts: applied to Z with linear sub-pixel interpolation.
    #     """
    
    #     assert len(self.volumes) > 0, "No volumes added."
    
    #     sy, sz, sx = self.volumes[0].shape
    
    #     # --- stack shift maps to (N, sy, sx); if any was missing, add() created zeros ---
    #     xmaps = np.asarray(self.xshifts)  # (N, sy, sx)
    #     ymaps = np.asarray(self.yshifts)  # (N, sy, sx)
    #     zmaps = np.asarray(self.zshifts)  # (N, sy, sx)
    
    #     # --- compute expansion symmetrically: allow negative & positive shifts ---
    #     # use nan-safe mins/maxes in case maps contain NaNs
    #     x_min = np.floor(np.nanmin(xmaps)).astype(int)
    #     x_max = np.ceil (np.nanmax(xmaps)).astype(int)
    #     y_min = np.floor(np.nanmin(ymaps)).astype(int)
    #     y_max = np.ceil (np.nanmax(ymaps)).astype(int)
    #     z_min = np.floor(np.nanmin(zmaps)).astype(int)
    #     z_max = np.ceil (np.nanmax(zmaps)).astype(int)
    
    #     # offsets bring everything into non-negative canvas indices
    #     off_x = -x_min
    #     off_y = -y_min
    #     off_z = -z_min
    
    #     expanded_x = sx + (x_max - x_min)
    #     expanded_y = sy + (y_max - y_min)
    #     expanded_z = sz + (z_max - z_min)
    
    #     self.corrected_volumes = []
    
    #     # --- iterate each volume with its shift maps ---
    #     for idx, (vol, xmap, ymap, zmap) in enumerate(zip(self.volumes, xmaps, ymaps, zmaps)):
    #         print(f"[RVS] Correcting volume {idx} on canvas "
    #               f"Y,X,Z=({expanded_y},{expanded_x},{expanded_z})")
    
    #         # accumulators to avoid overwrite artifacts
    #         sum_vol = np.zeros((expanded_y, expanded_z, expanded_x), dtype=np.float32)
    #         w_vol   = np.zeros((expanded_y, expanded_z, expanded_x), dtype=np.float32)
    
    #         # loop y,x; use rounded integer placement in XY
    #         for y in range(sy):
    #             # vectorize per-row if you want; this is clear & safe
    #             for x in range(sx):
    #                 dy = ymap[y, x]
    #                 dx = xmap[y, x]
    #                 dz = zmap[y, x]
    
    #                 if np.isnan(dy) or np.isnan(dx) or np.isnan(dz):
    #                     continue  # skip invalid shift entries
    
    #                 yy = int(np.round(y + dy + off_y))
    #                 xx = int(np.round(x + dx + off_x))
    
    #                 # bounds check in XY early
    #                 if yy < 0 or yy >= expanded_y or xx < 0 or xx >= expanded_x:
    #                     continue
    
    #                 # --- 1D subpixel shift along Z via linear splat ---
    #                 ascan = vol[y, :, x].astype(np.float32)  # (sz,)
    #                 # If there are NaNs in ascan, treat them as 0 and weight 0
    #                 valid = ~np.isnan(ascan)
    #                 if not np.any(valid):
    #                     continue
    
    #                 z_src = np.nonzero(valid)[0].astype(np.float32)
    #                 a_src = ascan[valid]
    
    #                 # destination (continuous) indices after shift and offset
    #                 t = z_src + float(dz) + float(off_z)        # (k,)
    #                 k0 = np.floor(t).astype(np.int64)           # lower bin
    #                 k1 = k0 + 1
    #                 w1 = (t - k0).astype(np.float32)            # fraction to upper
    #                 w0 = 1.0 - w1
    
    #                 # write (splat) into canvas with clipping
    #                 # lower bin
    #                 mask0 = (k0 >= 0) & (k0 < expanded_z)
    #                 if np.any(mask0):
    #                     sum_vol[yy, k0[mask0], xx] += a_src[mask0] * w0[mask0]
    #                     w_vol  [yy, k0[mask0], xx] += w0[mask0]
    
    #                 # upper bin
    #                 mask1 = (k1 >= 0) & (k1 < expanded_z)
    #                 if np.any(mask1):
    #                     sum_vol[yy, k1[mask1], xx] += a_src[mask1] * w1[mask1]
    #                     w_vol  [yy, k1[mask1], xx] += w1[mask1]
    
    #         # normalize where we wrote anything; leave others NaN
    #         corrected = np.full_like(sum_vol, np.nan, dtype=np.float32)
    #         nz = w_vol > 1e-6
    #         corrected[nz] = sum_vol[nz] / w_vol[nz]
    
    #         self.corrected_volumes.append(corrected)

    
    def correct_volumes(self):
        # Determine how large the rendered volume must be.
        # 1. Convert shifts lists to arrays
        # 2. Subtract minimum from each shift array
        # 3. Compute maximum value of shift in each dimension,
        #    and add this to the original volume dimensions

        sy, sz, sx = self.volumes[0].shape
        self.corrected_volumes = []
        
        self.xshifts = np.array(self.xshifts)
        self.yshifts = np.array(self.yshifts)
        self.zshifts = np.array(self.zshifts)
        
        # min-subtract all shifts
        self.xshifts = self.xshifts - np.min(self.xshifts)
        self.yshifts = self.yshifts - np.min(self.yshifts)
        self.zshifts = self.zshifts - np.min(self.zshifts)
        
        expanded_x = sx + math.ceil(np.max(self.xshifts))
        expanded_y = sy + math.ceil(np.max(self.yshifts))
        expanded_z = sz + math.ceil(np.max(self.zshifts))
        
        counter = 0
        for vol, x_shift_map, y_shift_map, z_shift_map in zip(self.volumes, self.xshifts, self.yshifts, self.zshifts):
            
            corrected_volume = np.full((expanded_y, expanded_z, expanded_x), np.nan, dtype=complex)
            print(f"Correcting volume {counter}")
            print(f"xshifts: {x_shift_map}\nyshifts: {y_shift_map}\nzshifts: {z_shift_map}")
            
            for y in range(sy):
                for x in range(sx):
                    yy = y + int(y_shift_map[y, x])
                    xx = x + int(x_shift_map[y, x])
                    zz = int(z_shift_map[y, x])

                    # Extract A-scan along z-axis
                    ascan = vol[y, :, x]  # shape: (sz,)
                    corrected_volume[yy, zz:zz+sz, xx] = ascan
                    # corrected_volume[yy, :sz, xx] = ascan
                    print(f"y = {y}, x = {x}")
            counter += 1      
                    
            # put the data into the correct locations here
            self.corrected_volumes.append(corrected_volume)

    #%% Synthetic data simulator
    # =========================
    # Synthetic data Simulator
    # =========================
    @staticmethod
    def simulate_synthetic_data(
        output_root='bscans_synthetic',
        info_dir='bscans_synthetic_info',
        N_scatterers=10000,
        sz_m=500e-6, sy_m=5e-4, sx_m=5e-4,
        L1=1e-6, L2=1.1e-6, N=256,
        lateral_resolution_sigma=5e-6,
        dt=1e-5,                      # 100 kHz, matches original
        x_scan_start=200e-6, x_scan_stop=300e-6, x_step=2e-6,
        y_scan_start=200e-6, y_scan_stop=300e-6, y_step=2e-6,
        n_volumes=2,
        motion_sigma_xyz=(3e-7, 3e-7, 3e-7),
        seed=None,
        save_png=True
    ):
        """
        Recreates generate_synthetic_data.py behavior and file outputs:
          - Writes <output_root>/%05d/bscan_%05d.npy B-scan tiles (complex)
          - Writes motion & scan arrays to <info_dir>/...npy
          - Optionally writes PNG previews (same colormap/limits)
        """
        # ---- embed minimal local types to avoid changing external imports ----
        import numpy as _np
        from matplotlib import pyplot as _plt

        if seed is not None:
            _np.random.seed(seed)

        # (Copied/trimmed structure from original script)  :contentReference[oaicite:3]{index=3}
        class _Scatterer:
            def __init__(self,z,y,x,r): self.z=z; self.y=y; self.x=x; self.r=r
            def lateral_distance(self,y,x): return _np.sqrt((self.x-x)**2+(self.y-y)**2)

        class _Sample:
            def __init__(self, sz_m, sy_m, sx_m):
                self.sz_m=sz_m; self.sy_m=sy_m; self.sx_m=sx_m; self.scatterers=[]
            def add_scatterer(self,z,y,x,r): self.scatterers.append(_Scatterer(z,y,x,r))
            def add_random_scatterer(self):
                self.add_scatterer(_np.random.rand()*self.sz_m,
                                   _np.random.rand()*self.sy_m,
                                   _np.random.rand()*self.sx_m,
                                   _np.random.rand())
            def randomize(self, N): [self.add_random_scatterer() for _ in range(N)]
            def move(self,dz,dy,dx):
                for s in self.scatterers: s.x+=dx; s.y+=dy; s.z+=dz
            def get_visible_scatterers(self,y,x,lim):
                return [s for s in self.scatterers if s.lateral_distance(y,x)<lim]
            
        class _OCT:
            def __init__(self):
                self.N = N
                self.L1 = L1; self.L2 = L2
                self.k1 = 2*_np.pi/self.L2; self.k2 = 2*_np.pi/self.L1
                self.k_arr = _np.linspace(self.k1,self.k2,self.N)
                self.k0 = (self.k1+self.k2)/2
                k_sigma = 1e5
                self.S = _np.exp(-(self.k_arr-self.k0)**2/(2*k_sigma**2))
                self.lateral_resolution_sigma = lateral_resolution_sigma
                self.r_r = 1.0
                self.simplified = True  # match original

            def spectral_scan(self, sample, y, x):
                vis = sample.get_visible_scatterers(y,x,6*self.lateral_resolution_sigma)
                signal = _np.zeros(self.N)
                if not self.simplified:
                    signal = signal + self.S*self.r_r
                for s in vis:
                    w = _np.exp(-((s.x-x)**2/(2*self.lateral_resolution_sigma**2)+
                                  (s.y-y)**2/(2*self.lateral_resolution_sigma**2)))
                    if not self.simplified:
                        signal = signal + w*self.S*s.r
                    signal = signal + w*2*self.S*_np.sqrt(self.r_r*s.r)*_np.cos(2*self.k_arr*s.z)
                return signal

        # --- instantiate and randomize sample & OCT ---
        samp = _Sample(sz_m, sy_m, sx_m); samp.randomize(N_scatterers)
        octsys = _OCT()

        # --- scan geometry & acquisition clock ---
        x_scan_range = _np.arange(x_scan_start, x_scan_stop, x_step)
        y_scan_range = _np.arange(y_scan_start, y_scan_stop, y_step)
        n_scans = len(x_scan_range)*len(y_scan_range)*n_volumes
        t_arr = _np.arange(0, n_scans*dt, dt)

        # --- motion traces, reference volume is zero-motion ---
        zsig, ysig, xsig = motion_sigma_xyz
        dz_trace = _np.random.randn(len(t_arr))*zsig
        dy_trace = _np.random.randn(len(t_arr))*ysig
        dx_trace = _np.random.randn(len(t_arr))*xsig
        scans_per_vol = len(x_scan_range)*len(y_scan_range)
        dz_trace[:scans_per_vol] = 0.0
        dy_trace[:scans_per_vol] = 0.0
        dx_trace[:scans_per_vol] = 0.0

        # --- persist metadata ---
        os.makedirs(info_dir, exist_ok=True)
        _np.save(os.path.join(info_dir,'dz_trace.npy'), dz_trace)
        _np.save(os.path.join(info_dir,'dy_trace.npy'), dy_trace)
        _np.save(os.path.join(info_dir,'dx_trace.npy'), dx_trace)
        _np.save(os.path.join(info_dir,'x_scan_range.npy'), x_scan_range)
        _np.save(os.path.join(info_dir,'y_scan_range.npy'), y_scan_range)

        # --- simulate & save volumes/bscans with identical shapes/limits ---
        os.makedirs(output_root, exist_ok=True)
        t_index = 0
        figure_made = False
        fig = None; ax = None

        for v in range(n_volumes):
            bscan_folder = os.path.join(output_root, f'{v:05d}')
            bscan_png_folder = os.path.join(output_root, f'{v:05d}_png')
            os.makedirs(bscan_folder, exist_ok=True)
            if save_png: os.makedirs(bscan_png_folder, exist_ok=True)

            for bscan_index, y in enumerate(y_scan_range):
                bscan = []
                for x in x_scan_range:
                    ss = octsys.spectral_scan(samp, y, x)
                    ascan = _np.fft.fft(ss)
                    bscan.append(ascan)
                    # apply motion step
                    samp.move(dz_trace[t_index], dy_trace[t_index], dx_trace[t_index])
                    t_index += 1

                bscan = _np.array(bscan).T
                bscan = bscan[:octsys.N//2, :]  # keep one conjugate side
                _np.save(os.path.join(bscan_folder, f'bscan_{bscan_index:05d}.npy'), bscan)

                if save_png:
                    if not figure_made:
                        bsy, bsx = bscan.shape
                        fig = _plt.figure(figsize=(bsx/50, bsy/50))
                        ax = fig.add_axes([0,0,1,1]); ax.set_xticks([]); ax.set_yticks([])
                        figure_made = True
                    ax.clear()
                    ax.imshow(_np.abs(bscan), cmap='gray', clim=(0,120))
                    _plt.savefig(os.path.join(bscan_png_folder, f'bscan_{bscan_index:05d}.png'), dpi=100)

        return dict(
            output_root=output_root, info_dir=info_dir,
            x_scan_range=x_scan_range, y_scan_range=y_scan_range,
            n_volumes=n_volumes
        )

    # --- registration of two synthetic volumes ---
    @staticmethod
    def register_synthetic_pair(
        data_root='bscans_synthetic',
        ref_index=0,
        target_index=1,
        sigma=5.0,
        compare_to_trace=True,
        info_folder='bscans_synthetic_info',
        samples_per_meter=5e5,
        show_plots=True
    ):
        """
        Recreates register_synthetic_volumes.py approach:
          - Loads volumes from disk
          - Row-wise Gaussian windowing; 3D phase correlation
          - Wrap-to-negative index handling (nxcswap)
          - Optional comparison to ground-truth motion traces
        Returns: (yshifts, zshifts, xshifts) as lists of ints (length = sy)
        """
        import numpy as _np
        from matplotlib import pyplot as _plt
        import glob as _glob

        def _load_volume(idx):
            folder = os.path.join(data_root, f'{idx:05d}')
            files = _glob.glob(os.path.join(folder, 'bscan*.npy'))
            files.sort()
            vol = [_np.load(f) for f in files]
            return _np.array(vol)  # (sy, sz, sx)

        ref = _load_volume(ref_index)
        target = _load_volume(target_index)

        sy, sz, sx = ref.shape
        y_arr = _np.arange(sy)

        def _nxcswap(a, N):  # identical behavior  :contentReference[oaicite:4]{index=4}
            return a - N if a > N//2 else a

        yshifts = []; zshifts = []; xshifts = []
        for y in range(sy):
            g = _np.exp(-(y_arr - y)**2 / (2*sigma**2))
            tar_t = _np.transpose(target, (1,2,0))
            tar_w = tar_t * g
            tar = _np.transpose(tar_w, (2,0,1))

            nxc3 = _np.abs(_np.fft.ifftn(_np.fft.fftn(tar) * _np.conj(_np.fft.fftn(ref))))
            yy, zz, xx = _np.unravel_index(_np.argmax(nxc3), nxc3.shape)

            yshifts.append(_nxcswap(yy, sy))
            zshifts.append(_nxcswap(zz, sz))
            xshifts.append(_nxcswap(xx, sx))

        if compare_to_trace:
            # Mirror the motion-trace comparison from the script  :contentReference[oaicite:5]{index=5}
            dx_trace = _np.load(os.path.join(info_folder,'dx_trace.npy'))
            dy_trace = _np.load(os.path.join(info_folder,'dy_trace.npy'))
            dz_trace = _np.load(os.path.join(info_folder,'dz_trace.npy'))

            n_ascans = sy * sx * 2  # two volumes used in original comparison
            n_bscans = sy * 2
            dx_trace = _np.cumsum(dx_trace[:n_ascans]) * samples_per_meter
            dy_trace = _np.cumsum(dy_trace[:n_ascans]) * samples_per_meter
            dz_trace = _np.cumsum(dz_trace[:n_ascans]) * samples_per_meter

            clock_ascan = _np.arange(n_ascans) * 10e-6
            clock_bscan = _np.arange(n_bscans) * 10e-6 * sx

            if show_plots:
                _plt.subplot(1,3,1)
                _plt.plot(clock_ascan, dx_trace, label='dx_trace')
                _plt.plot(clock_bscan[sy:], xshifts, label='x shift')  # align to 2nd volume start
                _plt.legend()
                _plt.subplot(1,3,2)
                _plt.plot(clock_ascan, dy_trace, label='dy_trace')
                _plt.plot(clock_bscan[sy:], yshifts, label='y shift')
                _plt.legend()
                _plt.subplot(1,3,3)
                _plt.plot(clock_ascan, dz_trace, label='dz_trace')
                _plt.plot(clock_bscan[sy:], zshifts, label='z shift')
                _plt.legend()
                _plt.show()

        return yshifts, zshifts, xshifts

    #%% rfunc
    # ===========
    # RFUNC WRAPS
    # ===========

    @staticmethod
    def _rfunc():
        """
        Lazy import so the module remains usable even if the helper file
        is moved or temporarily missing.
        """
        import importlib
        try:
            return importlib.import_module("ucd_registration_functions")
        except Exception as e:
            raise ImportError(
                "registration_functions.py not found or import failed. "
                "Make sure it sits next to ucd_registered_volume_series.py."
            ) from e

    # ---- loaders / cropping ----

    @classmethod
    def load_volume_from_folder(cls, folder: str, prefix: str = "bscan", *, crop: bool = False):
        """
        Wrapper over rfunc.get_volume / get_volume_and_crop (returns shape (slow, depth, fast)).
        """
        rfunc = cls._rfunc()
        vol = rfunc.get_volume(folder, prefix=prefix)            # (slow, depth, fast)
        return vol

    @classmethod
    def auto_crop_volume(cls, volume: np.ndarray):
        """
        Ask rfunc.auto_crop for z1/z2 based on z-profile, then slice.
        Expects (slow, depth, fast); returns cropped volume with same ordering.
        """
        rfunc = cls._rfunc()
        z1, z2 = rfunc.auto_crop(volume)
        return volume[:, z1:z2, :], (int(z1), int(z2))

    # ---- flattening (handles axis order differences automatically) ----

    @classmethod
    def flatten_volume_x(cls, volume_yzx: np.ndarray):
        """
        rfunc.flatten_volume expects (slow, fast, depth).
        Your pipeline uses (slow, depth, fast). We transpose to match,
        call rfunc, then transpose back.
        """
        rfunc = cls._rfunc()
        vol_yxz = np.transpose(volume_yzx, (0, 2, 1))        # (slow, fast, depth)
        flat_yxz = rfunc.flatten_volume(vol_yxz)              # internal shear-based x-flatten
        flat_yzx = np.transpose(flat_yxz, (0, 2, 1))          # back to (slow, depth, fast)
        return flat_yzx

    @staticmethod
    def make_x_flattener_from_bscan(bscan_zx: np.ndarray, min_shift=-30, max_shift=30, **kwargs):
        """
        Convenience wrapper: rfunc.get_xflattening_function works on a B-scan whose
        axis-0 is depth (z) and axis-1 is fast (x). Your typical bscan is (z, x),
        so you can pass it directly.
        Returns a callable(bscan_zx)->bscan_zx.
        """
        import registration_functions as rfunc
        return rfunc.get_xflattening_function(bscan_zx, min_shift=min_shift, max_shift=max_shift, **kwargs)

    # ---- correlation / upsampling / projections (simple pass-throughs) ----

    @staticmethod
    def nxc3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        import registration_functions as rfunc
        return rfunc.nxc3(a, b)

    @staticmethod
    def upsample_volume(volume: np.ndarray, factor: int) -> np.ndarray:
        import registration_functions as rfunc
        return rfunc.upsample(volume, factor)

    @staticmethod
    def project3(volume: np.ndarray, pfunc=np.nanmean):
        import registration_functions as rfunc
        return rfunc.project3(volume, pfunc=pfunc)

    @staticmethod
    def flythrough3(volume: np.ndarray, fps: int = 5):
        import registration_functions as rfunc
        return rfunc.flythrough3(volume, fps=fps)

    @staticmethod
    def generate_registration_manifest(folder: str, reference_label: str, upsample_factor: int = 2):
        import registration_functions as rfunc
        return rfunc.generate_registration_manifest(folder, reference_label, upsample_factor=upsample_factor)

    #%% Broadcasting
    # =========================
    # Broadcasting registration
    # =========================
    class _BroadcastReference:
        """
        Mirrors testing_broadcasting.ReferenceVolume:
        - caches 3D FFT of the reference volume
        - registers one 2D B-scan (slow row) to the 3D reference via broadcasting
        """
        def __init__(self, vol):
            self.vol = vol
            self.fref = np.fft.fftn(vol)
            self.n_slow, self.n_depth, self.n_fast = vol.shape
            # self.sy, self.sz, self.sx = vol.shape
            
        @staticmethod
        def _wrap_fix(p, size):
            # identical wrap convention used in testing_broadcasting.py (fix on z,x; not y)
            return p if p < size // 2 else p - size

        def register_bscan(self, target_bscan, poxc=True):
            """
            Register a single B-scan (shape: [depth, fast]) to the 3D reference.
            Uses broadcasting: FFT2(target) conj, multiply into fref, IFFTN, argmax.
            poxc=True => normalized cross-power (phase correlation) like original.
            """
            ftar = np.conj(np.fft.fft2(target_bscan))            # (depth, fast)
            # broadcast multiply across slow dimension:
            prod = self.fref * ftar                              # (slow, depth, fast)
            if poxc:
                prod = prod / (np.abs(prod) + 1e-12)
            # NOTE: testing_broadcasting computed ifftn(self.fref*ftar) again.
            # We keep the same behavior outcome by using 'prod' here.
            xc_arr = np.abs(np.fft.ifftn(self.fref * ftar))                  # (slow, depth, fast)

            # Peak & wrap the same way the script does (wrap z,x only)
            yp, zp, xp = np.unravel_index(np.argmax(xc_arr), xc_arr.shape)
            sy, sz, sx = xc_arr.shape
            zp = self._wrap_fix(zp, sz)
            xp = self._wrap_fix(xp, sx)
            return dict(dx=xp, dy=yp, dz=zp, xc=float(np.max(xc_arr)))
        
        # def register_bscan(self, bscan_zx, poxc=True, eps=1e-12):
        #     """
        #     bscan_zx: (sz, sx) for a single target row
        #     Returns: dict(dx, dy, dz, xc)
        #     """
        #     # 1) 2D FFT on (z,x) – this must match ref's last two axes
        #     Ftar = np.fft.fft2(bscan_zx)                # (sz, sx)
    
        #     # 2) Cross power w/ correct conjugation and normalization
        #     #    Broadcast Ftar over y to (sy, sz, sx)
        #     if poxc:
        #         num = (Ftar[None, :, :]) * np.conj(self.Fref)
        #         den = (np.abs(Ftar)[None, :, :] * np.abs(self.Fref)) + eps
        #         X = num / den
        #     else:
        #         X = (Ftar[None, :, :]) * np.conj(self.Fref)
    
        #     # 3) Inverse 3D FFT and pick the peak
        #     xc = np.fft.ifftn(X)
        #     a = np.abs(xc)
        #     yp, zp, xp = np.unravel_index(np.argmax(a), a.shape)
    
        #     # 4) Wrap-fix on ALL three axes (periodic correlation)
        #     yp = self._wrap_fix(yp, self.n_slow)
        #     zp = self._wrap_fix(zp, self.n_depth)
        #     xp = self._wrap_fix(xp, self.n_fast)
    
        #     return dict(dx=int(xp), dy=int(yp), dz=int(zp), xc=float(a[yp % self.n_slow, zp % self.n_depth, xp % self.n_fast]))

    @staticmethod
    def _tile_map_from_row_shifts(row_shifts, n_fast):
        """
        Turn a per-row (length = n_slow) 1D shift list/array into a (n_slow, n_fast) map
        by repeating across the fast axis, identical to:
            np.array([row_shifts for k in range(n_fast)]).T
        """
        row_shifts = np.asarray(row_shifts, dtype=float)
        return np.tile(row_shifts[:, None], (1, int(n_fast)))

    @classmethod
    def register_by_broadcasting(
        cls,
        ref_volume: np.ndarray,
        target_volumes: list,
        *,
        poxc: bool = True,
        plot: bool = True,
        save_dir: str | None = None,
    ):
        """
        Parameters
        ----------
        ref_volume : (n_slow, n_depth, n_fast) ndarray
            Reference volume.
        target_volumes : list[(n_slow, n_depth, n_fast) ndarray]
            Target volumes to register to the reference.
        poxc : bool
            Use phase-only cross-correlation normalization (matches script default).
        plot : bool
            Reproduce the four time-series plots (xc, dy, dz, dx).
        save_dir : str | None
            If provided, save xc arrays and x/y/z maps per target volume as .npy.

        Returns
        -------
        result : dict
            {
              "rvs": RegisteredVolumeSeries,
              "per_volume": [
                 {"xc": [...], "dx": [...], "dy": [...], "dz": [...],
                  "xmap": (n_slow,n_fast), "ymap": (...), "zmap": (...)}
              ],
              "corrected_volumes": list[np.ndarray],
            }
        """
        # 1) Instantiate RVS and add the reference (no shift maps needed)
        rvs = cls()
        rvs.add(ref_volume)

        # 2) Wrap the reference with a cached-FFT helper
        bref = cls._BroadcastReference(ref_volume)
        sy, sz, sx = ref_volume.shape

        per_volume_results = []

        # 3) Loop targets
        for vol_idx, target_vol in enumerate(target_volumes):
            xcmax_arr, x_shift_arr, y_shift_arr, z_shift_arr = [], [], [], []

            # per slow-scan (row): register target B-scan to the 3D reference
            for s in range(bref.n_slow):
                tar_bscan = target_vol[s, :, :]  # (depth, fast)
                res = bref.register_bscan(tar_bscan, poxc=poxc)
                xcmax_arr.append(res["xc"])
                x_shift_arr.append(res["dx"])
                # y shift convention: (peak y) minus current row index, same as script
                y_shift_arr.append(res["dy"] - s)
                z_shift_arr.append(res["dz"])

            # 4) Build maps by repeating row shifts across fast axis
            xmap = cls._tile_map_from_row_shifts(x_shift_arr, sx)
            ymap = cls._tile_map_from_row_shifts(y_shift_arr, sx)
            zmap = cls._tile_map_from_row_shifts(z_shift_arr, sx)

            # 5) Add to RVS
            rvs.add(target_vol, xmap, ymap, zmap)

            # 6) Optional: save arrays
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                np.save(os.path.join(save_dir, f"xcmax_arr_vol{vol_idx}.npy"), np.asarray(xcmax_arr))
                np.save(os.path.join(save_dir, f"x_map_vol{vol_idx}.npy"), xmap)
                np.save(os.path.join(save_dir, f"y_map_vol{vol_idx}.npy"), ymap)
                np.save(os.path.join(save_dir, f"z_map_vol{vol_idx}.npy"), zmap)

            # 7) Optional plots (one figure per target, four stacked traces)
            if plot:
                fig = plt.figure(figsize=(9, 7))
                ax = fig.add_subplot(4, 1, 1); ax.plot(xcmax_arr); ax.set_ylabel("xc")
                ax = fig.add_subplot(4, 1, 2); ax.plot(y_shift_arr); ax.set_ylabel("dy")
                ax = fig.add_subplot(4, 1, 3); ax.plot(z_shift_arr); ax.set_ylabel("dz")
                ax = fig.add_subplot(4, 1, 4); ax.plot(x_shift_arr); ax.set_ylabel("dx"); ax.set_xlabel("row")
                fig.suptitle(f"Broadcast registration traces (target vol #{vol_idx})")
                plt.tight_layout()
                plt.show()

            per_volume_results.append(
                dict(
                    xc=xcmax_arr, dx=x_shift_arr, dy=y_shift_arr, dz=z_shift_arr,
                    xmap=xmap, ymap=ymap, zmap=zmap
                )
            )

        # 8) Correct volumes through the class’ existing API
        rvs.correct_volumes()

        return dict(
            rvs=rvs,
            per_volume=per_volume_results,
            corrected_volumes=getattr(rvs, "corrected_volumes", []),
        )

    @classmethod
    def register_by_broadcasting_from_root(
        cls,
        root: str,
        indices: list[int] | None = None,
        *,
        prefix: str = "bscan_",  # keep flexible; original used rfunc.get_volume_and_crop
        loader=None,             # optional callable: (folder) -> volume ndarray
        **kwargs
    ):
        """
        Convenience: load volumes from numbered subfolders under `root`,
        then call `register_by_broadcasting`. Keep behavior flexible; you can
        pass a custom `loader` that replicates `rfunc.get_volume_and_crop`.
        """
        # discover folders 00000, 00001, ...
        subfolders = sorted([p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)])
        if indices is None:
            indices = list(range(len(subfolders)))
        folders = [subfolders[i] for i in indices]

        def _default_loader(folder):
            # Minimal loader: stack all *.npy B-scans in lexical order into (n_slow, n_depth, n_fast).
            files = sorted(glob.glob(os.path.join(folder, "*.npy")))
            bscans = [np.load(f) for f in files]
            vol = np.stack(bscans, axis=0)  # (n_slow, n_depth, n_fast)
            return vol

        _loader = loader if loader is not None else _default_loader

        ref_vol = _loader(folders[0])
        target_vols = [ _loader(f) for f in folders ]  # include ref first, like the script
        return cls.register_by_broadcasting(ref_vol, target_vols, **kwargs)
    
    #%% End-to-End registration pipeline
    # ===========
    # NEAT PIPELINE
    # ===========
    
    @classmethod
    def build_and_register_from_root(
        cls,
        root: str,
        *,
        prefix: str = "bscan",
        indices: list[int] | None = None,
        crop: bool = False,
        flatten_x: bool = False,
        upsample_factor: int | None = None,
        use_broadcasting: bool = True,
        plot: bool = True,
        save_dir: str | None = None,
        **broadcast_kwargs,
    ):
        """
        One high-level call that UTILIZES rfunc helpers, then uses class registration.
    
        Steps:
        1) Load volumes (optionally auto-crop at load-time).
        2) Optional x-flatten (rfunc shear-based).
        3) Optional upsample (rfunc).
        4) Register via your broadcasting method (or other RVS method ).
        5) Correct volumes with RVS API; return a structured dict.
    
        Returns:
            {
              "rvs": RegisteredVolumeSeries,
              "folders": [...],
              "volumes": [...],              # (possibly cropped/flattened/upsampled)
              "registration": {...},         # result of register_by_broadcasting(...)
              "z_crop": (z1, z2) or None,
            }
        """
        # Discover subfolders
        subfolders = sorted([p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)])
        if indices is None:
            indices = list(range(len(subfolders)))
        folders = [subfolders[i] for i in indices]
        if len(folders) == 0:
            raise ValueError(f"No subfolders found under: {root}")
    
        # Load, optional crop/flatten/upsample
        vols = []
        z_crop = None
        for f in folders:
            vol = cls.load_volume_from_folder(f, prefix=prefix, crop=False)  # (slow, depth, fast)
            if crop and z_crop is None:
                # Record the first crop window for reference
                vol, z_crop = cls.auto_crop_volume(vol)
            elif crop:
                # For consistent depth, apply the same crop to others
                z1, z2 = z_crop
                vol = vol[:, z1:z2, :]
    
            if flatten_x:
                vol = cls.flatten_volume_x(vol)
    
            if upsample_factor and upsample_factor > 1:
                vol = cls.upsample_volume(vol, upsample_factor)
    
            vols.append(vol)
    
        # Register + Correct
        if use_broadcasting:
            reg = cls.register_by_broadcasting(
                ref_volume=vols[0],
                target_volumes=vols,        # like your script: include ref first
                plot=plot,
                save_dir=save_dir,
                **broadcast_kwargs
            )
            rvs = reg["rvs"]
        else:
            # You can swap in any other RVS registration you’ve added.
            rvs = cls()
            rvs.add(vols[0])
            for v in vols[1:]:
                rvs.add(v)  # (add shift maps if your other method computes them)
            rvs.correct_volumes()
            reg = {"rvs": rvs, "per_volume": [], "corrected_volumes": getattr(rvs, "corrected_volumes", [])}
    
        return {
            "rvs": rvs,
            "folders": folders,
            "volumes": vols,
            "registration": reg,
            "z_crop": z_crop,
        }

