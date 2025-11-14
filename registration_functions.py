import numpy as np
from matplotlib import pyplot as plt
import sys,os,glob
import scipy.interpolate as spi
import scipy.signal as sps
#import inspect
#from matplotlib.widgets import Button, Slider, SpanSelector, Cursor
import pandas as pd
import json

def load_dict(fn):
    with open(fn,'r') as fid:
        s = fid.read()
        d = json.loads(s)
    return d

def save_dict(fn,d):
    s = json.dumps(d)
    with open(fn,'w') as fid:
        fid.write(s)

def shear(bscan,max_roll):
    out = np.zeros(bscan.shape,dtype=complex)
    roll_vec = np.linspace(0,max_roll,bscan.shape[1])
    roll_vec = np.round(roll_vec).astype(int)
    for k in range(bscan.shape[1]):
        out[:,k] = np.roll(bscan[:,k],roll_vec[k])
    return out

def get_xflattening_function(bscan,min_shift=-30,max_shift=30,diagnostics=False,do_plot=False):
    shift_range = range(min_shift,max_shift) # [-20, -19, ..... 19, 20]
    peaks = np.zeros(len(shift_range)) # [0, 0, ..... 0, 0]
    profs = []
    for idx,shift in enumerate(shift_range): # iterate through [-20, -19, ..... 19, 20]
        temp = shear(bscan,shift) # shear by -20, then -19, then -18...
        prof = np.mean(np.abs(temp),axis=1) # compute the lateral median
        profs.append(prof)
        peaks[idx] = np.max(prof) # replace the 0 in peaks with whatever the max value is of prof
    # now, find the location of the highest value in peaks, and use that index to find the optimal shift
    optimal_shift = shift_range[np.argmax(peaks)]
    profs = np.array(profs)
    if do_plot:
        fig = plt.figure(figsize=(6,8))
        ax = fig.subplots(2,1)
        ax[0].imshow(profs.T,aspect='auto')
        ax[1].plot(shift_range,peaks)
        ax[1].set_xlabel('max shear')
        ax[1].set_ylabel('max profile peak')
        plt.show()
    return lambda bscan: shear(bscan,optimal_shift)


def flatten_volume(volume):
    n_slow,n_fast,n_depth = volume.shape
    temp = np.mean(np.abs(volume[n_slow//2-3:n_slow//2+3,:,:]),axis=0)
    f = get_xflattening_function(temp)
    flattened = []
    for s in range(n_slow):
        flattened.append(f(volume[s,:,:]))

    flattened = np.array(flattened,dtype=complex)
    targets = np.mean(np.abs(flattened),axis=2)
    plt.figure()
    plt.imshow(targets)
    
    reference = np.mean(targets[n_slow//2-3:n_slow//2+3,:],axis=0)
    shifts = [xcorr(t,reference) for t in targets]
    shifts = sps.medfilt(shifts,5)
    out = []
    out = [np.roll(b,s,axis=0) for b,s in zip(flattened,shifts)]
    out = np.array(out)
    return out

def generate_registration_manifest(folder, reference_label, upsample_factor = 2):
    outfn = folder + '_registration_manifest.json'
    flist = glob.glob(os.path.join(folder,'*'))
    flist.sort()
    d = {}
    d['reference'] = reference_label
    d['targets'] = flist
    d['upsample_factor'] = upsample_factor
    save_dict(outfn,d)

def get_peaks(prof,count=np.inf):
    # return the COUNT brightest maxima in prof
    left = prof[:-2]
    center = prof[1:-1]
    right = prof[2:]
    peaks = np.where((center>=left)*(center>=right))[0]+1
    if peaks.shape[0] < count:
        print('only %s peaks exists, decrease count to %s'%(peaks.shape[0],peaks.shape[0]))
        count = peaks.shape[0]
    peak_vals = prof[peaks]
    thresh = sorted(peak_vals)[-count]
    peaks = peaks[np.where(prof[peaks]>=thresh)]
    return list(peaks)

def auto_crop(img):
    zprof = np.nanmean(img,axis=(0,2))
    plt.plot(zprof)
    points = get_peaks(zprof, count=2)
    if points[0] < 100:
        img = img[:,100:,:]
        zprof = np.nanmean(img,axis=(0,2))
        plt.figure()
        plt.plot(zprof)
        plt.show()
        points = get_peaks(zprof, count=2)
        bscan_z1 = 100 #cfg.z_padding_px
        bscan_z2 = 240
        # bscan_z1 = np.min(points) - 10 + 100 #cfg.z_padding_px
        # bscan_z2 = np.min(points) + 70 + 100
    else:
        bscan_z1 = np.min(points) - 70 #cfg.z_padding_px
        bscan_z2 = np.min(points) + 70
    return bscan_z1, bscan_z2

def get_volume(folder,prefix='bscan'):
    flist = glob.glob(os.path.join(folder,'%s*.npy'%prefix))
    flist.sort()
    
    vol = [np.load(f) for f in flist]
    vol = np.array(vol)
    
    return vol#[:10,:20,:30]

def get_volume_and_crop(folder,prefix='bscan'):
    flist = glob.glob(os.path.join(folder,'%s*.npy'%prefix))
    flist.sort()
    
    vol = [np.load(f) for f in flist]
    vol = np.array(vol)
    
    # show image projection
    plt.figure()
    # plt.imshow(np.abs(vol[1,:,:]), cmap="gray", aspect='auto')
    plt.imshow(np.abs(np.nanmean(vol,axis=0)), cmap="gray", aspect='auto')
    plt.title("original image")
    plt.show()
    
    vol_z1, vol_z2 = auto_crop(vol)
    vol_crop = vol[:,vol_z1:vol_z2,:]
    
    # show cropped image projection
    plt.figure()
    # plt.imshow(np.abs(vol[1,:,:]), cmap="gray", aspect='auto')
    plt.imshow(np.abs(np.nanmean(vol_crop,axis=0)), cmap="gray", aspect='auto')
    plt.title("cropped image")
    plt.show()
    
    return vol_crop#[:10,:20,:30]

def upsample(vol,factor):
    return vol.repeat(factor,axis=0).repeat(factor,axis=1).repeat(factor,axis=2)


def nxc3(a,b):
    prod = np.fft.fftn(a)*np.conj(np.fft.fftn(b))
    aprod = np.abs(prod)+1e-16
    #prod = prod/aprod
    out = np.abs(np.fft.ifftn(prod))
    return out

def flythrough3(a,fps=5):
    for k in range(len(a.shape)):
        for d in range(a.shape[0]):
            frame = a[d,:,:]
            if np.all(np.isnan(frame)):
                continue
            plt.cla()
            plt.imshow(a[d,:,:])
            plt.title('dim %d frame %d'%(k,d))
            plt.pause(1.0/fps)
        a = np.transpose(a,[1,2,0])
        

def project3(a,pfunc=np.nanmean):
    plt.figure(figsize=(9,3))
    plt.subplot(1,3,1)
    plt.imshow(pfunc(a,axis=0),cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(pfunc(a,axis=1),cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(pfunc(a,axis=2),cmap='gray')
    plt.show()


