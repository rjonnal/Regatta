import numpy as np
from matplotlib import pyplot as plt
import config as cfg
import sys,os,glob
import scipy.interpolate as spi
import scipy.signal as sps
import inspect
from matplotlib.widgets import Button, Slider, SpanSelector, Cursor
import pandas as pd
import inspect
from skimage.registration import phase_cross_correlation

# %% FILE PATHS AND NAMES
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter import filedialog

# %%
def save3Dastif(matrix3d, outfn):
    import numpy as np
    import tifffile as tiff    
    # Save as .tif without compression
    tiff.imwrite(outfn, matrix3d, dtype=np.float64, compression=None)    
    print(f"3D matrix saved as {outfn} without compression.")
    
def saveConfig(filename):
    import config
    # Get all attributes from config.py
    parameters = {name: value for name, value in vars(config).items() if not name.startswith('__')}
    # Save parameters to a .txt file
    output_file = filename.replace('.bin','') +'_config.txt'
    with open(output_file, "w") as f:
        for key, value in parameters.items():
            f.write(f"{key} = {value}\n")    
    print(f"Parameters saved to {output_file}")
    
def choose_file():
    # Create a Tkinter root window (hidden)
    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring the dialog to the front

    # Open file dialog and get the selected file path
    file_path = askopenfilename(title="Select a File")   
    
    return file_path


def choose_folder():
    # Create a hidden Tkinter window
    root = Tk()
    root.withdraw()  # Hide the root window    
    root.attributes('-topmost', True)  # Bring the dialog to the front
    root.update()
    # Open the folder selection dialog
    folder_path = filedialog.askdirectory(title="Select a Folder")
    # Destroy the root window to free resources
    root.destroy()
    if folder_path:
        print(f"Selected folder: {folder_path}")
    else:
        print("No folder selected.")
    
    return folder_path


# %%
def autocrop(volume):
    prof = get_profile(volume)
    # plt.figure()
    # plt.plot(prof)
    # plt.show()
    # sys.exit()
    z1 = 0
    z2 = z1 + cfg.desired_depth_px
    if len(prof) > cfg.desired_depth_px:
        # totals = []
        # while z2<=len(prof):
        #     totals.append(np.std(prof[z1:z2]))
        #     z1+=1
        #     z2+=1
        ax_index = np.argmax(prof)
        shift = round(ax_index - cfg.desired_depth_px/2)
        z1 = np.max([0, int(z1 + shift)])
        z2 = int(z1+cfg.desired_depth_px)
    print('Z1-'+str(z1))
    return volume[:,z1:z2,:]

def dB(arr):
    return 20*np.log10(np.abs(arr))

def phase_to_nm(phase):
    return phase/(4*np.pi*1.38)*cfg.L0

def nm_to_phase(nm):
    return nm*(4*np.pi*1.38)/cfg.L0

def xcorr2d(tar,ref):
    cftar = np.conj(np.fft.fft2(tar))
    fref = np.fft.fft2(ref)
    nxc = np.abs(np.fft.ifft2(fref*cftar/np.abs(fref)/np.abs(cftar)))
    sy,sx = nxc.shape
    py,px = np.unravel_index(np.argmax(nxc),nxc.shape)
    if py>sy//2:
        py = py-sy
    if px>sx//2:
        px = px-sx
    return py,px,nxc
    
    peak_index = np.argmax(nxc)
    if peak_index>len(tar)//2:
        peak_index = peak_index-len(tar)
    return peak_index

def xcorr(tar,ref):
    nxc = np.abs(np.fft.ifft(np.fft.fft(ref)*np.conj(np.fft.fft(tar))))
    peak_index = np.argmax(nxc)
    if peak_index>len(tar)//2:
        peak_index = peak_index-len(tar)
    return peak_index

def flatten_volume(volume):
    n_slow,n_fast,n_depth = volume.shape
    #get on optimal shear shift, then apply to all Bscans
    temp = np.mean(np.abs(volume[n_slow//2-3:n_slow//2+3,:,:]),axis=0) #-3, +3
    f = get_xflattening_function(temp)
    flattened = []
    for s in range(n_slow):
        flattened.append(f(volume[s,:,:]))
    
    # ##get on optimal shear shift for each Bscan
    # flattened = []
    # deltay = 5
    # for s in np.arange(0,n_slow,deltay):
    #     temp = np.mean(np.abs(volume[s:s+deltay,:,:]),axis=0) #-3, +3
    #     f = get_xflattening_function(temp)
    #     for y in np.arange(s, s+deltay):
    #         # print('y%d'%y)
    #         flattened.append(f(volume[y,:,:]))

    flattened = np.array(flattened,dtype=complex)
    targets = np.mean(np.abs(flattened),axis=2)
    # plt.figure()
    # plt.imshow(targets)
    
    reference = np.mean(targets[n_slow//2-3:n_slow//2+3,:],axis=0)
    shifts = [xcorr(t,reference) for t in targets]
    shifts = sps.medfilt(shifts,5)
    out = []
    out = [np.roll(b,s,axis=0) for b,s in zip(flattened,shifts)]
    out = np.array(out)
    return out

def flatten_volume0(volume):
    avol = np.abs(volume)
    n_slow,n_depth,n_fast = avol.shape
    kernel = np.zeros(avol.shape)
    kwidth = 9
    kernel[n_slow//2:n_slow//2+4,n_depth//2,n_fast//2:n_fast//2+kwidth] = 1.0/kwidth
    favol = np.fft.fftn(avol)#,s=(n_slow*2,n_depth*2,n_fast*2))
    fkernel = np.fft.fftn(kernel)#,s=(n_slow*2,n_depth*2,n_fast*2))
    savol = np.abs(np.fft.ifftn(favol*fkernel))#[n_slow//2:n_slow//2+n_slow,n_depth//2:n_depth//2+n_depth,:n_fast//2:n_fast//2+n_fast]
    savol = np.fft.fftshift(savol)
    ref = savol[n_slow//2,:,n_fast//2]
    savol = np.transpose(savol,[0,2,1])
    fref = np.fft.fft(ref)
    fsavol = np.fft.fft(savol,axis=2)
    xc = np.fft.ifft(fref*np.conj(fsavol),axis=2)
    xcmax = np.argmax(np.abs(xc),axis=2)
    xcmax[np.where(xcmax>n_depth//2)] = xcmax[np.where(xcmax>n_depth//2)]-n_depth
    xcmax = sps.medfilt(xcmax,35).astype(int) #35
    for s in range(n_slow):
        for f in range(n_fast):
            volume[s,:,f] = np.roll(volume[s,:,f],xcmax[s,f])
    return volume


def get_projections(vol,show=True):
    avol = np.abs(vol)
    slow_projection = np.mean(avol,axis=0)
    fast_projection = np.mean(avol,axis=2).T
    depth_projection = np.mean(avol,axis=1)
    if show:
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(dB(slow_projection),aspect='auto',interpolation='none',cmap='gray')
        plt.title('projection in\nslow scan dimension')
        plt.colorbar()
        plt.subplot(1,3,2)
        plt.imshow(dB(fast_projection),aspect='auto',interpolation='none',cmap='gray')
        plt.title('projection in\nfast scan dimension')
        plt.colorbar()
        plt.subplot(1,3,3)
        plt.imshow(dB(depth_projection),aspect='auto',interpolation='none',cmap='gray')
        plt.title('projection in\ndepth dimension')
        plt.colorbar()
        plt.tight_layout()
    return slow_projection,fast_projection,depth_projection

def get_profile(volume):
    return np.mean(np.mean(np.abs(volume),axis=2),axis=0)

def z_align(volume,reference):
    target = get_profile(volume)
    shift = xcorr(reference,target)
    volume = np.roll(volume,-shift,axis=1)
    if False:
        shifted = get_profile(volume)
        plt.figure()
        plt.plot(reference, label='from ref vol')
        plt.plot(target, label='target')
        plt.plot(shifted, label='after Z align')
        plt.title('depth profile, shift =' + str(shift))
        plt.xlim(50,250)
        plt.legend()
        plt.show()
        # sys.exit()
    return volume

def autocorrelation_2d(a):
    sy,sx = a.shape
    fa = np.fft.fft2(a)
    afa = np.abs(fa)
    return np.fft.fftshift(np.abs(np.fft.ifft2(fa*np.conj(fa))))[sy//2-10:sy//2+10,
                                                                 sx//2-10:sx//2+10]

def get_peaks(prof,count=np.inf):
    # return the COUNT brightest maxima in prof
    left = prof[:-2]
    center = prof[1:-1]
    right = prof[2:]
    peaks = np.where((center>=left)*(center>=right))[0]+1
    peak_vals = prof[peaks]
    thresh = sorted(peak_vals)[-count]
    peaks = peaks[np.where(prof[peaks]>=thresh)]
    return list(peaks)


def get_peaks_ui(prof):

    # Create the figure and the line that we will manipulate
    layer_markers = cfg.layer_markers
    points = get_peaks(prof,count=2)
    fig,ax = plt.subplots(1,1)

    def f():
        ax.clear()
        ax.plot(prof)
        for pt in points:
            #print('drawing at %d'%pt)
            plt.axvline(pt,color='r')
        ax.set_title('%d layer_indices: %s'%(len(points),points))
        
    f()
    def on_click(event):
        x = int(round(event.xdata))
        if x in points:
            points.remove(x)
        else:
            points.append(x)
        f()

        fig.canvas.draw_idle()
        
    
    cursor = Cursor(ax, useblit=True, color='red', linewidth=2, vertOn=True, horizOn=False)
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    return sorted(points)



def get_peaks_2d(im):

    mind = 100
    
    right = im[2:,:]
    hcenter = im[1:-1,:]
    left = im[:-2,:]
    hpeaks = np.where(np.logical_and(hcenter>right+mind,hcenter>left+mind))
    hpoints = list(zip(hpeaks[0]+1,hpeaks[1]))

    top = im[:,:-2]
    vcenter = im[:,1:-1]
    bottom = im[:,2:]
    vpeaks = np.where(np.logical_and(vcenter>top+mind,vcenter>bottom+mind))
    vpoints = list(zip(vpeaks[0],vpeaks[1]+1))

    points = []

    blank = np.zeros(im.shape)
    for hpoint in hpoints:
        blank[hpoint[0],hpoint[1]]+=1
    for vpoint in vpoints:
        blank[vpoint[0],vpoint[1]]+=1

    points = np.where(blank==2)
    points = list(zip(points[0],points[1]))
    return points

def get_peaks_2d_ui(im):

    # Create the figure and the line that we will manipulate
    points = get_peaks_2d(im)
    fig,ax = plt.subplots(1,1,figsize=(10,10*cfg.bscans_per_volume/172))
    clim = np.percentile(im,(1,99))
    def f():
        ax.clear()
        ax.imshow(im,cmap='gray',clim=clim, aspect='auto')
        for pt in points:
            #print('drawing at %d'%pt)
            plt.plot(pt[1],pt[0],'r.',alpha=0.5)
        ax.set_title('%d points'%len(points),fontsize=32)       
    f()
    def on_click(event):
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if (y,x) in points:
            points.remove((y,x))
        else:
            points.append((y,x))
        f()
        fig.canvas.draw_idle()
    cursor = Cursor(ax, useblit=True, color='red', linewidth=2, vertOn=True, horizOn=False)
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()
    plt.show()

    return points,fig



# def autocrop(volume):
#     prof = get_profile(volume)
#     # plt.figure()
#     # plt.plot(prof)
#     # plt.show()
#     # sys.exit()
#     z1 = 0
#     z2 = cfg.desired_depth_px
#     totals = []
#     while z2<=len(prof):
#         totals.append(np.std(prof[z1:z2]))
#         z1+=1
#         z2+=1
#     z1 = np.argmax(totals)
#     z2 = z1+cfg.desired_depth_px
#     return volume[:,z1:z2,:]

def sharpness(im):
    """Image sharpness"""
    return np.sum(im**2)/(np.sum(im)**2)

def contrast(im):
    """Image contrast"""
    return (np.max(im)-np.min(im))/(np.max(im)+np.min(im))

def immax(im):
    """Image max"""
    return np.max(im)

def remove_artifacts(bscan):
    """ Work in progress."""
    prof = np.var(np.abs(bscan),axis=1)/np.mean(np.abs(bscan),axis=1)
    print('%0.1e,%0.1e,%0.1e,%0.1e'%(np.max(prof),np.min(prof),np.mean(prof),np.median(prof)))
    return bscan

def check_file_size(filename):
    file_size = os.path.getsize(filename)
    expected_file_size = cfg.n_volumes*cfg.bscans_per_volume*cfg.ascans_per_bscan*cfg.samples_per_ascan*cfg.bytes_per_sample
    assert (file_size-expected_file_size)/2/cfg.ascans_per_bscan/cfg.n_volumes-cfg.extra_samples_per_bscan == 0

def centroid(vec):
    x = np.arange(len(vec))
    return np.sum(vec*x)/np.sum(vec)

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
        ax[0].set_xlabel('shift range [pixel]')
        ax[0].set_ylabel('Depth [pixel]')
        ax[0].set_title('averaged A-scan profile')
        ax[1].plot(shift_range,peaks)
        ax[1].plot(optimal_shift, np.max(peaks),'r*')
        ax[1].set_xlabel('shift range [pixel]')
        ax[1].set_ylabel('max profile peak')
        plt.show()
    # print(optimal_shift)
    return lambda bscan: shear(bscan,optimal_shift)


def get_cells_filename(filename):
    return os.path.join(get_org_folder(filename),'cells.npy')

def get_coordinates_filename(filename):
    return os.path.join(get_org_folder(filename),'yx_coordinates.npy')

def get_depth_profile(cell):
    return np.nanmean(np.nanmean(np.abs(cell),axis=2),axis=0)



def get_shear(bscan,deg=1):
    sy,sx = bscan.shape
    temp = np.zeros(bscan.shape)
    temp[np.where(bscan>np.mean(bscan))] = bscan[np.where(bscan>np.mean(bscan))]
    coms = []
    for x in range(sx):
        #coms.append(centroid(temp[:,x]))
        coms.append(np.argmax(temp[:,x]))
    p = np.polyfit(np.arange(sx),coms,deg=deg)
    pval = np.polyval(p,np.arange(sx))
    pval = pval - np.mean(pval)
    pval = np.round(pval).astype(int)
    return pval
    
def get_spectra_folder(filename):
    folder = filename.replace('.bin','')+'_spectra'
    os.makedirs(folder,exist_ok=True)
    return folder

def get_spectra_filename(volume_index,bscan_index):
    return 'spectra_%05d_%05d.npy'%(volume_index,bscan_index)

def get_bscan_folder(filename):
    folder = filename.replace('.bin','')+'_bscans'#+'_zeropad'+str(cfg.zeropad_fold)
    os.makedirs(folder,exist_ok=True)
    return folder

def get_registered_bscan_folder(filename):
    folder = filename.replace('.bin','')+'_registered_bscans'
    os.makedirs(folder,exist_ok=True)
    return folder

def get_projections_folder(filename):
    folder = filename.replace('.bin','')+'_projections'
    os.makedirs(folder,exist_ok=True)
    return folder

def get_org_folder(filename):
    folder = filename.replace('.bin','')+'_org'
    os.makedirs(folder,exist_ok=True)
    return folder

def get_projections_filename(filename,volume_index):
    return os.path.join(get_projections_folder(filename),'projection_%05d.npy'%volume_index)

def get_projections_png_filename(filename,volume_index):
    return os.path.join(get_projections_folder(filename),'projection_%05d.png'%volume_index)

def get_registered_projections_folder(filename):
    folder = filename.replace('.bin','')+'_registered_projections'
    os.makedirs(folder,exist_ok=True)
    return folder

def get_registered_projections_filename(filename,volume_index):
    return os.path.join(get_registered_projections_folder(filename),'projection_%05d.npy'%volume_index)

def get_registered_projections_png_filename(filename,volume_index):
    return os.path.join(get_registered_projections_folder(filename),'projection_%05d.png'%volume_index)

def get_volume_folder(filename,volume_index):
    folder = os.path.join(get_bscan_folder(filename),'%05d'%volume_index)
    os.makedirs(folder,exist_ok=True)
    return folder

def get_registered_volume_folder(filename,volume_index):
    folder = os.path.join(get_registered_bscan_folder(filename),'%05d'%volume_index)
    os.makedirs(folder,exist_ok=True)
    return folder

def get_volume(filename,volume_index):
    flist = glob.glob(os.path.join(get_volume_folder(filename,volume_index),'bscan*.npy'))
    flist.sort()
    vol = [np.load(f) for f in flist]
    vol = np.array(vol)
    # vol = vol[:,250:450,:]
    return vol

def get_registered_volume(filename,volume_index):
    flist = glob.glob(os.path.join(get_registered_volume_folder(filename,volume_index),'bscan*.npy'))
    if len(flist)==0:
        return None
    flist.sort()
    vol = [np.load(f) for f in flist]
    vol = np.array(vol)
    return vol

def save_volume(filename,volume,volume_index):
    for bidx in range(volume.shape[0]):
        print('Writing bscan to %s.'%get_bscan_filename(filename,volume_index,bidx))
        np.save(get_bscan_filename(filename,volume_index,bidx),volume[bidx,:,:])

def save_registered_volume(filename,volume,volume_index):
    for bidx in range(volume.shape[0]):
        print('Writing bscan to %s.'%get_registered_bscan_filename(filename,volume_index,bidx))
        np.save(get_registered_bscan_filename(filename,volume_index,bidx),volume[bidx,:,:])

        
def get_bscan_filename(filename,volume_index,bscan_index):
    return os.path.join(get_volume_folder(filename,volume_index),'bscan_%05d.npy'%bscan_index)

def get_registered_bscan_filename(filename,volume_index,bscan_index):
    return os.path.join(get_registered_volume_folder(filename,volume_index),'bscan_%05d.npy'%bscan_index)

def get_desine_LUT_filename():
    return 'desine_LUT.txt'

# def desine(bscan,scan_radians=np.pi):
#     desine_LUT_filename = get_desine_LUT_filename()
#     N = cfg.ascans_per_bscan
#     try:
#         lut = np.loadtxt(desine_LUT_filename).astype(int)
#     except FileNotFoundError:
#         amp_deg = cfg.resonant_scanner_amplitude_deg
#         freq = cfg.resonant_scanner_frequency
#         sampled_locations = np.cos(np.linspace(scan_radians,0,N)-0.1)*amp_deg/2.0+amp_deg/2.0
#         interpolated_locations = np.linspace(0,amp_deg,N)

#         lut = []
#         for k in range(N):
#             test = interpolated_locations[k]
#             winner = np.argmin(np.abs(sampled_locations-test))
#             lut.append(int(winner))

#         lut = np.array(lut,dtype=int)
#         np.savetxt(desine_LUT_filename,lut,fmt='%d')
    
#     desined = []
#     for idx in range(N):
#         desined.append(bscan[:,lut[idx]])
#     desined = np.array(desined).T
#     return desined

def desine(bscan,scan_radians=cfg.default_scan_range_radians,scan_offset=cfg.default_scan_offset_radians):
    desine_LUT_filename = get_desine_LUT_filename()
    N = cfg.ascans_per_bscan
    try:
        lut = np.loadtxt(desine_LUT_filename+'foo').astype(int)        
    except:
        amp_deg = cfg.resonant_scanner_amplitude_deg
        freq = cfg.resonant_scanner_frequency
        sampled_locations = np.cos(np.linspace(scan_radians,0,N)+scan_offset)*amp_deg/2.0+amp_deg/2.0
        interpolated_locations = np.linspace(0,amp_deg,N)
    
        lut = []
        for k in range(N):
            test = interpolated_locations[k]
            winner = np.argmin(np.abs(sampled_locations-test))
            lut.append(int(winner))
    
        lut = np.array(lut,dtype=int)
    # np.savetxt(desine_LUT_filename,lut,fmt='%d')
    
    desined = []
    for idx in range(N):
        desined.append(bscan[:,lut[idx]])
    desined = np.array(desined).T
    return desined


def load_spectra(filename,volume_index,bscan_index,flip=True):
    # print(np.ravel(np.fromfile(filename,offset=cfg.initial_skip*cfg.bytes_per_sample,count=10,dtype=cfg.dtype)))
    # sys.exit()
    file_size = os.path.getsize(filename)
    n_samples_total = file_size/cfg.bytes_per_sample
    n_samples_per_volume = n_samples_total/cfg.n_volumes
    n_samples_per_bscan = n_samples_per_volume/cfg.bscans_per_volume
    
    bscan_stride = cfg.ascans_per_bscan*cfg.samples_per_ascan
    volume_stride = n_samples_per_bscan*cfg.bscans_per_volume

    offset = cfg.bytes_per_sample*(volume_stride*volume_index + n_samples_per_bscan*bscan_index + cfg.initial_skip)
    count = cfg.ascans_per_bscan*cfg.samples_per_ascan
    
    try:
        assert int(offset)==offset
        assert int(count)==count
    except Exception as e:
        print(e)
        print('Calculated data parameters are non-integers.')
        
    count = int(count)
    offset = int(offset)
    dat = np.fromfile(filename,count=count,offset=offset,dtype=cfg.dtype)
    dat = np.reshape(dat,(cfg.ascans_per_bscan,cfg.samples_per_ascan)).T
    if flip:
        dat = dat[::-1,:]
    return dat.astype(float)





def get_dispersion_filename(filename):
    return filename.replace('.bin','')+'_dispersion_coefficients.txt'

def get_dispersion_coefficients(filename):
    dispersion_filename = get_dispersion_filename(filename)
    try:
        coefs = np.loadtxt(dispersion_filename)
    except FileNotFoundError:
        print('Warning: no dispersion coefficients found. Using [0.0, 0.0].')
        coefs = np.array([0.0,0.0])
    return coefs

def get_dc_filename(filename):
    return filename.replace('.bin','')+'_dc.npy'


def get_dc(filename,stride=1):
    dc_filename = get_dc_filename(filename)
    try:
        dc = np.load(dc_filename)
    except FileNotFoundError as fnfe:
        
        bpv = cfg.bscans_per_volume
        all_spectra = []
        for k in range(0,bpv,stride):
            spectra = load_spectra(filename,0,k) #volume 0
            all_spectra.append(spectra)

        all_spectra = np.array(all_spectra)
        dc = np.mean(all_spectra,axis=0)
        np.save(dc_filename,dc)
    return dc

def dc_subtract(spectra,dc):
    #return (spectra.T-dc).T
    return spectra #- dc

def window(spectra,w):
    return (spectra.T*w).T

def dispersion_compensate(spectra,k,coefficients):
    # If all coefficients are 0, return the spectra w/o further computation:
    if not any(coefficients):
        return spectra

    # the coefficients passed into this function are just the 3rd and 2nd order ones; we
    # add zeros so that we can use convenience functions like np.polyval that handle the
    # algebra; the input coefficients are [dc3,dc2], either a list or numpy array;
    # cast as a list to be on the safe side.
    coefs = list(coefficients) + [0.0,0.0]

    # now coefs is a 4-item list: [dc3,dc2,0.0,0.0]
    

    # if we want to avoid using polyval, we can explicitly evaluate the polynomial:
    # evaluate our polynomial on index x; if we do it this way, we need not append
    # zeroes to coefs above
    # phase_polynomial = coefs[0]*k**3 + coefs[1]*k**2

    # actually it's simpler to use polyval, which is why we appended the zeros to
    # the input coefficients--polyval infers the order of the polynomial from the
    # number of values in the list/array:
    phase_polynomial = np.polyval(coefs,k)    
    dc_polynomial = np.polyfit(k,phase_polynomial,1)
    dc_ramp = np.polyval(dc_polynomial,k)
    
    if False:
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(k, phase_polynomial, label='phase_polynomial')
        plt.plot(k, dc_ramp,label='dc in phase_polynomial')
        plt.xlabel('k value=$(2\pi/\lambda)$')
        plt.ylabel('phase')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(k, phase_polynomial-dc_ramp)
        plt.xlabel('k value=$(2\pi/\lambda)$')
        plt.ylabel('phase')
        plt.title('phase delay induced by the dispersion')
        plt.tight_layout()
    
    phase_polynomial = phase_polynomial - dc_ramp   
    # define the phasor and multiply by spectra using broadcasting:
    dechirping_phasor = np.exp(-1j*phase_polynomial)
    # spectra = sps.hilbert(spectra,axis=0)
    dechirped = (spectra.T*dechirping_phasor).T
    # dechirped = np.real(dechirped)
    return dechirped


def load_mapping_indices():
    mapping_filename = cfg.mapping_index_filename
    if mapping_filename.lower().endswith('.ind'):
        with open(mapping_filename,'rb') as fid:
            dat = np.fromfile(fid,dtype=np.float32,offset=8)
    if mapping_filename.lower().endswith('.txt'):
        dat = np.loadtxt(mapping_filename)
    return dat



class Diagnostics:

    def __init__(self,tag,limit=20):
        if tag.find('_bscans')>-1:
            tag = tag.replace('_bscans/','')
        if tag.find('.unp')>-1:
            tag = tag.replace('.unp','')
        if tag.find('.bin')>-1:
            tag = tag.replace('.bin','')
            
        self.folder = tag+'_diagnostics'
        os.makedirs(self.folder,exist_ok=True)
        self.limit = limit
        self.dpi = 300
        self.figures = {}
        self.labels = {}
        self.counts = {}
        self.done = []
        self.current_figure = None

    def log(self,title,header,data,fmt,clobber):
        print(title)
        print(header)
        print(fmt%data)
        
    def save(self,figure_handle=None,ignore_limit=False):

        if figure_handle is None:
            figure_handle = self.current_figure
        label = self.labels[figure_handle]
        
        if label in self.done:
            return
        
        subfolder = os.path.join(self.folder,label)
        os.makedirs(subfolder,exist_ok=True)
        index = self.counts[label]

        if index<self.limit or ignore_limit:
            outfn = os.path.join(subfolder,'%s_%05d.png'%(label,index))
            plt.figure(label)
            plt.suptitle('%s_%05d'%(label,index))
            plt.savefig(outfn,dpi=self.dpi)
            #plt.show()
            self.counts[label]+=1
        else:
            self.done.append(label)
        #plt.close(figure_handle.number)
            

    def figure(self,figsize=(6,6*cfg.bscans_per_volume/172),dpi=100,label=None):
        if label is None:
            label = inspect.currentframe().f_back.f_code.co_name
            
        subfolder = os.path.join(self.folder,label)
        if not label in self.counts.keys():
            self.counts[label] = 0
            os.makedirs(subfolder,exist_ok=True)
        fig = plt.figure(label)
        self.labels[fig] = label
        fig.clear()
        fig.set_size_inches(figsize[0],figsize[1], forward=True)
        #out = plt.figure(figsize=figsize,dpi=dpi)
        self.current_figure = fig
        return fig


class Bscans:

    def __init__(self,filename,diagnostics=True):
        #check_file_size(filename)
        self.filename = filename
        if diagnostics:
            self.diagnostics = Diagnostics(self.filename)
        else:
            self.diagnostics = False
            
        self.L0 = cfg.L0
        self.delta_L = cfg.delta_L
        self.L1 = self.L0 - self.delta_L/2.0
        self.L2 = self.L0 + self.delta_L/2.0
        self.k1 = 2*np.pi/self.L1
        self.k2 = 2*np.pi/self.L2
        
        self.k_out = np.linspace(self.k2,self.k1,cfg.samples_per_ascan) # 2304? 
        self.k_in = load_mapping_indices() #in a decreasing order
        self.window = np.hanning(cfg.samples_per_ascan)
        # self.window = np.exp(-(np.linspace(-1, 1, cfg.samples_per_ascan) ** 2) /0.707)
        self.dc = get_dc(filename)
        self.bscan_dict = {}
        self.dispersion_coefficients = get_dispersion_coefficients(filename)
        self.bscans_per_volume = cfg.bscans_per_volume
        self.zeropad_fold = cfg.zeropad_fold
        
    def get_bscan(self,volume_index,bscan_index,diagnostics=False):
        spectra = load_spectra(self.filename,volume_index,bscan_index)
        bscan = self.spectra_to_bscan(spectra,self.dispersion_coefficients)
        return bscan

    def k_linearize(self,spectra):
        for f in range(spectra.shape[1]):
            k_interpolator = spi.interp1d(self.k_in,spectra[:,f],kind='nearest',fill_value='extrapolate')
            spectra[:,f] = k_interpolator(self.k_out)
        return spectra

    def spectra_zeropad(self, spectra):
        spectra_padded = np.pad(spectra, pad_width=[(0, spectra.shape[0]*(self.zeropad_fold-1)), (0, 0)], mode='constant')
        return spectra_padded
        
    def spectra_to_bscan(self,spectra,dispersion_coefficients=[0.0,0.0]):
        #DC, ksample, dispersion, window (old)
        #DC, window, ksample, dispersion
        spectra0 = spectra
        spectra1 = dc_subtract(spectra0,self.dc)
        spectra2 = window(spectra1,self.window)
        spectra3 = self.k_linearize(spectra2) 
        if cfg.do_dispersion_compensation:
            spectra4 = dispersion_compensate(spectra3,self.k_out,dispersion_coefficients)
        else:
            spectra4 = spectra3
        # spectra2 = window(spectra4,self.window)
        f = lambda s: 20*np.log10(np.abs(np.fft.fft(s,axis=0)))[cfg.bscan_z1:cfg.bscan_z2,:]
        def prof(spectra):
            out = np.mean(f(spectra),axis=1)
            return out
        if self.diagnostics:
            climbscan = cfg.dispersion_dB_clims #[75,105]
            fig = self.diagnostics.figure(figsize=(12,16))
            out = fig.subplots(5,3)
            out[0][0].imshow(spectra0,aspect='auto')
            spectra0_padded = self.spectra_zeropad(spectra0)
            out[0][1].imshow(f(spectra0_padded),cmap='gray',clim=climbscan,aspect='auto')
            out[0][2].plot(prof(spectra0))
            out[0][2].grid()    
            out[0][0].set_ylabel('raw spectra')

            out[1][0].imshow(spectra1,aspect='auto')
            spectra1_padded = self.spectra_zeropad(spectra1)
            out[1][1].imshow(f(spectra1_padded),cmap='gray',clim=climbscan,aspect='auto')
            out[1][2].plot(prof(spectra1_padded))
            out[1][2].grid()    
            out[1][0].set_ylabel('dc subtracted')

            out[2][0].imshow(spectra2,aspect='auto')
            spectra2_padded = self.spectra_zeropad(spectra2)
            out[2][1].imshow(f(spectra2_padded),cmap='gray',clim=climbscan,aspect='auto')
            out[2][2].plot(prof(spectra2_padded))
            out[2][2].grid()    
            out[2][0].set_ylabel('windowed')

            out[3][0].imshow(spectra3,aspect='auto')
            spectra3_padded = self.spectra_zeropad(spectra3)
            out[3][1].imshow(f(spectra3_padded),cmap='gray',clim=climbscan,aspect='auto')
            out[3][2].plot(prof(spectra3_padded))
            out[3][2].grid()    
            out[3][0].set_ylabel('k linearized')

            out[4][0].imshow(np.abs(spectra4),aspect='auto')
            spectra4_padded = self.spectra_zeropad(spectra4)
            out[4][1].imshow(f(spectra4_padded),cmap='gray',clim=climbscan,aspect='auto')
            out[4][2].plot(prof(spectra4_padded))
            out[4][2].grid()              
            out[4][0].set_ylabel('dispersion compensated')
            self.diagnostics.save()
        spectra4_padded = self.spectra_zeropad(spectra4)
        bscan = np.fft.fft(spectra4_padded,axis=0)
        # bscan = np.fft.fft(spectra4,axis=0)
        bscan = bscan[cfg.bscan_z1:cfg.bscan_z2,:]
        bscan = desine(bscan)
        return bscan


def strip_register(folder,reference_index,sigma=cfg.default_sigma,show=False):
    
    flist = glob.glob(os.path.join(folder,'projection*.npy'))
    flist.sort()
    ref_filename = flist[reference_index]
    
    ref = np.load(ref_filename)
    plt.figure(figsize=(10,10*cfg.bscans_per_volume/172))
    plt.imshow(ref,cmap='gray',aspect = 'auto')
    plt.title('reference frame, Close to continue. CTRL-C to quit.')
    plt.show()
    clim = np.percentile(ref,(0.5,99.9))

    #precompute ref fft
    fref = np.fft.fft2(ref)
    afref = np.abs(fref)

    output_header = ['xshift','yshift','ycoord','correlation','reference_index','reference_filename']
    output_rows = []
    if show:
        plt.figure(figsize=(10,10*cfg.bscans_per_volume/172))
        
    for f in flist:
        output_filename = f.replace('.npy','')+'_strip_registration.csv'
        tar = np.load(f)
        sy,sx = tar.shape
        output_rows = []
        print('Registering %s.'%f,end='')
        for row in range(sy):
            if not show:
                print('.',end='')
                sys.stdout.flush()
            y = np.arange(sy)-row
            g = np.exp(-(y**2)/(2*sigma**2))
            #g = g/np.sum(g)
            strip = (g*tar.T).T
            fstrip = np.conj(np.fft.fft2(strip))
            afstrip = np.abs(fstrip)
            
            xc = np.abs(np.fft.ifft2(fref*fstrip/afref/afstrip))
            py,px = np.unravel_index(np.argmax(xc),xc.shape)
            xcmax = np.max(xc)
            yshift,xshift = py,px
            if yshift>sy//2:
                yshift = yshift - sy
            if xshift>sx//2:
                xshift = xshift - sx
            
            ##########################################
            # TODO: apply the same shift to the whole image
            # shift_values, error, diffphase = phase_cross_correlation(ref, tar, upsample_factor=100)
            # yshift,xshift = shift_values
            ############################################
            #print(xshift,yshift,xcmax)

            xplot = np.array([0,sx,sx,0,0])
            yplot = np.array([row-2*sigma,row-2*sigma,row+2*sigma,row+2*sigma,row-2*sigma])

            rxplot = xplot+xshift
            ryplot = yplot+yshift

            # we want to record the coordinates in the reference frame where this strip goes
            # the x location is given by xshift
            # the y location is given by row+yshift
            output_rows.append([xshift,yshift,row+yshift,xcmax,reference_index,ref_filename])

            if show:
                plt.clf()
                plt.subplot(2,3,1)
                plt.imshow(ref,cmap='gray',clim=clim)
                plt.plot(rxplot,ryplot,'y-')
                plt.title('ref')
                plt.xlim((0,sx))
                plt.ylim((sy,0))
                plt.subplot(2,3,4)                
                plt.imshow(tar,cmap='gray',clim=clim)
                plt.title('target')
                plt.plot(xplot,yplot,'y-')
                plt.xlim((0,sx))
                plt.ylim((sy,0))
                plt.subplot(2,3,3)
                plt.imshow(strip,cmap='gray',clim=clim)
                plt.title('smoothed strip')
                plt.subplot(2,3,6)
                plt.imshow(20*np.log10(np.abs(xc)))
                plt.plot(px,py,'rs',markersize=10,markerfacecolor='none')
                plt.title('xcorr map')
                plt.subplot(2,3,2)
                plt.title('zoom ref')
                plt.imshow(ref,cmap='gray',clim=clim,aspect='auto')
                plt.plot(rxplot,ryplot,'y-')
                plt.xlim(rxplot[:2])
                plt.ylim(ryplot[-2:])
                plt.subplot(2,3,5)
                plt.title('zoom target')
                plt.imshow(tar,cmap='gray',clim=clim,aspect='auto')
                plt.plot(xplot,yplot,'y-')
                plt.xlim(xplot[:2])
                plt.ylim(yplot[-2:])
                plt.suptitle('yshift,xshift=%d, %d'%(yshift,xshift))
                plt.tight_layout()
                plt.pause(.000001)
            
        df = pd.DataFrame(output_rows,columns=output_header)
        df.to_csv(output_filename)
        print()

def get_reference_index(filename):
    projection_folder = get_projections_folder(filename)
    csv_list = glob.glob(os.path.join(projection_folder,'*_strip_registration.csv'))
    df = pd.read_csv(csv_list[0])
    return int(np.array(df['reference_index'])[0])

def register_and_average_projections(filename):
    projection_folder = get_projections_folder(filename)
    csv_list = glob.glob(os.path.join(projection_folder,'*_strip_registration.csv'))
    csv_list.sort()
    projection_list = glob.glob(os.path.join(projection_folder,'projection*.npy'))
    projection_list.sort()

    test = np.load(projection_list[0])
    sy,sx = test.shape

    canvas = np.zeros((sy*2,sx*2))
    counter = np.ones((sy*2,sx*2))*1e-16
   
    # projection_mean2 = []
    oy = sy//2
    ox = sx//2

    test = pd.read_csv(csv_list[0])
    
    ref_fn = str(test['reference_filename'][0])
    ref_fn = os.path.join(projection_folder, os.path.split(ref_fn)[-1])
    ref = np.load(ref_fn)
    
    canvas[oy:oy+sy,ox:ox+sx] = ref
    counter[oy:oy+sy,ox:ox+sx] = 1.0
    
    for cf,pf in zip(csv_list,projection_list):
        df = pd.read_csv(cf)
        # check that the reference filenames are correct
        reffn_list = list(df['reference_filename'])
        matches = [r==ref_fn for r in reffn_list]
        try:
            assert all(matches)
        except AssertionError as ae:
            print(ae)
            print('Problem with reference filenames:')
            print(reffn_list)

        try:
            assert len(df)==ref.shape[0]
        except AssertionError as ae:
            print(ae)
            print('Disagreement betweein window height %d and length of csv file %d.'%(ref.shape[0],len(df)))

        xshift_vec = np.array(df['xshift'])
        yshift_vec = np.array(df['yshift'])
        registered_projs = np.zeros((sy*2,sx*2))
        target = np.load(pf)
        for idx,row in df.iterrows():
            target_row = target[idx,:]
            xshift = int(row['xshift'])
            yshift = int(row['yshift'])
            corr = float(row['correlation'])
            #TODO
            if corr<cfg.minimum_correlation:
                continue
            if np.sqrt(xshift**2+yshift**2)>cfg.max_displacement:
                continue
            if np.abs(xshift)>ox or np.abs(yshift)>oy:
                continue
            canvas[idx+oy+yshift,ox+xshift:ox+xshift+sx]+=target_row
            counter[idx+oy+yshift,ox+xshift:ox+xshift+sx]+=1
            registered_projs[idx+oy+yshift,ox+xshift:ox+xshift+sx]=target_row #add by YC
        projection_mean = canvas/counter # projection_mean=registered_projs
        # projection_mean2.append((registered_projs))
        
        plt.figure(figsize=(10,10*cfg.bscans_per_volume/172))
        plt.cla()
        plt.imshow(registered_projs[oy:oy+sy,ox:ox+sx],cmap='gray',aspect='auto',clim=np.percentile(ref,(1,99)))
        plt.title(os.path.basename(pf))
        plt.pause(.1)
        
    projection_mean = projection_mean[oy:oy+sy,ox:ox+sx]
    # projection_mean2 = np.array(projection_mean2)
    org_folder = get_org_folder(filename)
    
    fig = plt.figure(figsize=(10,10*cfg.bscans_per_volume/172))
    ax = fig.add_axes([0,0,1,1])
    ax.set_xticks([])
    ax.set_yticks([])    
    plt.imshow(projection_mean,cmap='gray',aspect='auto',clim=np.percentile(ref,(1,99)))
    plt.title('registered_average_projection')
    # fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1])
    # ax.set_xticks([])
    # ax.set_yticks([])  
    # plt.imshow(np.mean(projection_mean2[:,oy:oy+sy,ox:ox+sx],0),cmap='gray',aspect='auto',clim=np.percentile(ref,(0.2,95)))
    # plt.title('Map B-scans by overwrite-registered_volume')
    
    plt.savefig(os.path.join(org_folder,'registered_average_projection.png'),dpi=cfg.projections_dpi)
    plt.show()
    np.save(os.path.join(org_folder,'registered_average_projection.npy'),projection_mean)


def register_and_average_volumes(filename, method='update'):
    bscan_folder = get_bscan_folder(filename)
    csv_list = glob.glob(os.path.join(get_projections_folder(filename),'*_strip_registration.csv'))
    csv_list.sort()
    volume_list = glob.glob(os.path.join(bscan_folder,'*'))
    volume_list.sort()

    test_volume = get_volume(filename,0)
    sy,sz,sx = test_volume.shape

    canvas = np.zeros((sy*2,sz,sx*2),dtype=complex)
    counter = np.ones((sy*2,sz,sx*2))*1e-16
    
    oy = sy//2
    ox = sx//2

    volume_indices = range(cfg.n_volumes)
    
    for cf,vidx in zip(csv_list,volume_indices):
        df = pd.read_csv(cf)
        try:
            assert len(df)==test_volume.shape[0]
        except AssertionError as ae:
            print(ae)
            print('Disagreement between NBscan %d and length of csv file %d.'%(test_volume.shape[0],len(df)))

        xshift_vec = np.array(df['xshift'])
        yshift_vec = np.array(df['yshift'])

        target = get_volume(filename,vidx) #slow, depth, fast
        registered_volume = np.zeros(canvas.shape,dtype=complex)
        canvas2 = (np.zeros((sy*2,sz,sx*2),dtype=complex))
        counter2 = np.ones((sy*2,sz,sx*2))*1e-16
        for idx,row in df.iterrows():
            target_bscan = target[idx,:,:]
            xshift = int(row['xshift'])
            yshift = int(row['yshift'])
            corr = float(row['correlation'])
            # TODO
            if corr<cfg.minimum_correlation:
                continue
            if np.sqrt(xshift**2+yshift**2)>cfg.max_displacement:
                continue
            if np.abs(xshift)>ox or np.abs(yshift)>oy:
                continue
            canvas[idx+oy+yshift,:,ox+xshift:ox+xshift+sx]+=target_bscan #sum up for all vols
            canvas2[idx+oy+yshift,:,ox+xshift:ox+xshift+sx]+=(target_bscan) #sum up for one vol
            registered_volume[idx+oy+yshift,:,ox+xshift:ox+xshift+sx]=target_bscan #saved version, overwrite
            counter[idx+oy+yshift,:,ox+xshift:ox+xshift+sx]+=1 
            counter2[idx+oy+yshift,:,ox+xshift:ox+xshift+sx]+=1
            
        volume_mean = canvas/counter
        volume_mean2 = canvas2/counter2
        volume_mean2 = volume_mean2[oy:oy+sy,:,ox:ox+sx]        
        registered_volume = registered_volume[oy:oy+sy,:,ox:ox+sx]
        if method=='update':
            save_registered_volume(filename,registered_volume,vidx)
        if method == 'average':
            save_registered_volume(filename,volume_mean2,vidx)
        
        if False:
            plt.figure(figsize=(10,10))
            plt.cla()
            plt.imshow(np.mean(np.abs(volume_mean),axis=1),cmap='gray',aspect='auto')
            plt.pause(.1)
            plt.figure(figsize=(10,10))
            plt.cla()
            plt.imshow(np.mean(np.abs(registered_volume),axis=1),cmap='gray',aspect='auto')

    volume_mean = volume_mean[oy:oy+sy,:,ox:ox+sx]
    # np.save(get_org_folder('volume_average.npy'),volume_mean)


    
if __name__=='__main__':
    fn = sys.argv[1]
    bscans = Bscans(fn)
    
    for v in range(1):
        for b in range(4):
            print(v,b)
            bscan = bscans.get_bscan(v,b)
            plt.cla()
            plt.imshow(np.abs(bscan))
            plt.show()
