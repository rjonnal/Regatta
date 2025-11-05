import registration_functions as rfunc
import numpy as np
from matplotlib import pyplot as plt
import sys,os,glob
import registered_volume_series
import functions as blobf
import config as cfg

def auto_crop(img):
    zprof = np.nanmean(img,axis=(0,2))
    plt.plot(zprof)
    points = blobf.get_peaks_ui(zprof)
    bscan_z1 = np.min(points) - 70 #cfg.z_padding_px
    bscan_z2 = np.min(points) + 70
    return bscan_z1, bscan_z2

# root = r'E:\Registration\data_Ravi\bscans_synthetic'
# root = r'E:\Registration\data_Ravi\bscans_aooct'
root = r'F:\Registration\Fast\locNS05_b6_bscans'
# root = r'E:\Registration\data_Ravi\bscans_oct_widefield_CHOIR_0052'

f0 = os.path.join(root,'00000')
f1 = os.path.join(root,'00001')
f2 = os.path.join(root,'00002')
f3 = os.path.join(root,'00003')
f4 = os.path.join(root,'00004')
f5 = os.path.join(root,'00005')

v0 = rfunc.get_volume_and_crop(f0,prefix='')
v1 = rfunc.get_volume_and_crop(f1,prefix='')
v2 = rfunc.get_volume_and_crop(f2,prefix='')
v3 = rfunc.get_volume_and_crop(f3,prefix='')
v4 = rfunc.get_volume_and_crop(f4,prefix='')
v5 = rfunc.get_volume_and_crop(f5,prefix='')

n_slow,n_depth,n_fast = v0.shape

plt.figure()
plt.imshow(np.abs(v3[100,:, :]))
plt.show()

def fix(p,s):
    if p<s//2:
        return p
    else:
        return p-s


class ReferenceVolume:
    """A class representing the reference volume. This is handy because it
    can store the 3D FFT of the volume such that it doesn't have to be recomputed
    for each B-scan in the target volume."""
    
    def __init__(self,vol):
        self.vol = vol
        self.fref = np.fft.fftn(vol)
        self.n_slow,self.n_depth,self.n_fast = self.vol.shape
        
    def register(self,target_bscan,poxc=True):
        """Register a single bscan to the reference volume via broadcasting"""
        ftar = np.conj(np.fft.fft2(target_bscan))
        prod = self.fref*ftar
        if poxc:
            prod = prod/np.abs(prod)
        xc_arr = np.abs(np.fft.ifftn(self.fref*ftar))

        # plt.figure()
        # for k in range(3):
        #     plt.subplot(1,3,k+1)
        #     plt.imshow(np.max(xc_arr,axis=k))
        # plt.show()
        xc = np.max(xc_arr)

        yp,zp,xp = np.unravel_index(np.argmax(xc_arr),xc_arr.shape)
        sy,sz,sx = xc_arr.shape
        #yp = fix(yp,sy)
        zp = fix(zp,sz)
        xp = fix(xp,sx)

        result = {}
        result['dx'] = xp
        result['dy'] = yp
        result['dz'] = zp
        result['xc'] = xc
        print(result)
        return result


def register_volumes(refvol,target_volumes_list):

    rvs = registered_volume_series.RegisteredVolumeSeries()
    rvs.add(refvol)
    
    refvol = ReferenceVolume(refvol)
    
    count = 0
    for target_volume in target_volumes_list:
        xcmax_arr = []
        y_shift_arr = []
        z_shift_arr = []
        x_shift_arr = []
        for s in range(refvol.n_slow):
            tar = target_volume[s,:,:]
            res = refvol.register(tar)
            xcmax_arr.append(res['xc'])
            x_shift_arr.append(res['dx'])
            y_shift_arr.append(res['dy']-s)
            z_shift_arr.append(res['dz'])

        # filter shifts using xc here, later

        xmap = np.array([x_shift_arr for k in range(refvol.n_fast)]).T
        ymap = np.array([y_shift_arr for k in range(refvol.n_fast)]).T
        zmap = np.array([z_shift_arr for k in range(refvol.n_fast)]).T

        rvs.add(target_volume,xmap,ymap,zmap)
        
        np.save(fr"F:\Registration\Fast\boradcasting_10232025\xcmax_arr_vol{count}.npy", xcmax_arr)
        np.save(fr"F:\Registration\Fast\boradcasting_10232025\x_map_vol{count}.npy", xmap)
        np.save(fr"F:\Registration\Fast\boradcasting_10232025\y_map_vol{count}.npy", ymap)
        np.save(fr"F:\Registration\Fast\boradcasting_10232025\z_map_vol{count}.npy", zmap)
        
        plt.subplot(4,1,1)
        plt.plot(xcmax_arr)
        plt.ylabel('xc')
        plt.subplot(4,1,2)
        plt.plot(y_shift_arr)
        plt.ylabel('dy')
        plt.subplot(4,1,3)
        plt.plot(z_shift_arr)
        plt.ylabel('dz')
        plt.subplot(4,1,4)
        plt.plot(x_shift_arr)
        plt.ylabel('dx')

        plt.show()
        count += 1
        # sys.exit()
    rvs.correct_volumes()
    
    count = 0
    for cv in rvs.corrected_volumes:
        print(cv)
        # np.save(rf"F:\Registration\Fast\boradcasting_10232025\rvs_corrected_vol{count}", cv)
        count += 1
    
    # np.save(r"F:\Registration\Fast\boradcasting_10232025\raw_vol5", v5)
    
    cv = np.nanmean(np.abs(np.array(rvs.corrected_volumes)),axis=0)
    
    rfunc.project3(cv,pfunc=np.nanmax)
    rfunc.flythrough3(cv)

            
register_volumes(v0,[v0, v1, v2, v3, v4, v5])

