# Fast4Dreg.py package
# Intended Command Line usage: python Fast4Dreg.py image.tif output_dir
# dependencies:
#%%
# Imports
# Essentials
import numpy as np
import dask
import dask.array as da
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, AffineTransform
# from scipy.ndimage import affine_transform, rotate 
from dask_image.ndinterp import affine_transform, rotate
from dask.diagnostics import ProgressBar
# Utility
from tqdm import tqdm
import tifffile 
import sys
import os

def get_xy_drift(_data, ref_channel, _registration_mode):
    
    if _registration_mode == 'mean': 
        xy_movie = da.average(_data[ref_channel], axis = 1).compute()
    elif _registration_mode == 'max': 
        xy_movie = da.max(_data[ref_channel], axis = 1).compute()
    elif _registration_mode == 'median': 
        xy_movie = da.max(_data[ref_channel], axis = 1).compute()
        
    # correct XY, relative to first frame:
    print('Determining drift in XY')
    shifts, error, phasediff = [], [], []
    for t in tqdm(range(len(xy_movie)-1)):        

        s, e, p = phase_cross_correlation(xy_movie[t],
                                            xy_movie[t+1], 
                                            normalization=None) 
        
        

        shifts.append(s)
        error.append(e)
        phasediff.append(p)
        
    shifts_xy = np.cumsum(np.array(shifts), axis = 0)
    shifts_xy = shifts_xy.tolist()
    shifts_xy.insert(0,[0,0])
    shifts_xy = np.asarray(shifts_xy)
    
    plt.title('XY-Drift')
    plt.plot(shifts_xy[:,0], label = 'x')
    plt.plot(shifts_xy[:,1], label = 'y')
    plt.legend()
    plt.xlabel('Timesteps')
    plt.ylabel('Drift in pixel')
    plt.savefig('XY-Drift.svg')
    plt.clf()
    return shifts_xy

def get_z_drift(_data, ref_channel, _registration_mode):
    
    if _registration_mode == 'mean': 
        z_movie = da.average(da.swapaxes(_data[ref_channel], 2,1), axis = 1).compute() 
    elif _registration_mode == 'max': 
        z_movie = da.max(da.swapaxes(_data[ref_channel], 2,1), axis = 1).compute() 
    elif _registration_mode == 'median':
        z_movie = da.median(da.swapaxes(_data[ref_channel], 2,1), axis = 1).compute() 
        
        
    # correct XY, relative to first frame:
    print('Determining drift in Z')
    shifts, error, phasediff = [], [], []
    for t in tqdm(range(len(z_movie)-1)):        

        s, e, p = phase_cross_correlation(  z_movie[t],
                                            z_movie[t+1], 
                                            normalization=None) 
        
        

        shifts.append(s)
        error.append(e)
        phasediff.append(p)
        
    shifts_z = np.cumsum(np.array(shifts), axis = 0)
    for i in shifts_z:
        i[1] = 0

    shifts_z = shifts_z.tolist()
    shifts_z.insert(0,[0,0])
    shifts_z = np.asarray(shifts_z)


    plt.title('Z-Drift')
    plt.plot(shifts_z[:,0], label = 'z')
    plt.legend()
    plt.xlabel('Timesteps')
    plt.ylabel('Drift in pixel')
    plt.savefig('Z-Drift.svg')
    plt.clf()

    return shifts_z

def translate_stack(_image, _shift): 

    # define warp matrix:
    tform = AffineTransform(translation=-_shift)

    # construct shifted stack
    _image_out = []
    
    for c,C in enumerate(_image):
        _out_stack = []
        for z,Z in enumerate(C):
            
            # here is the delayed function that allows to scatter everthing nicely across workers
            _out_stack.append(affine_transform(Z, np.array(tform)))
        
        # format arrays for proper export
        _out_stack = da.stack(_out_stack)
        _image_out.append(_out_stack)

    return da.stack(_image_out)


def apply_xy_drift(_data, xy_drift):

    # Expectes _data to have the following shape:
    # np.shape_data = (channel, time, z, y, x)

    # Swap axes to iterate over time which aligns with chunks
    _data = da.swapaxes(_data, 0,1)
    
    print('Scheduling tasks...')
    _translated_data = []
    for t,T in tqdm(enumerate(_data)):
        _translated_data.append(translate_stack(T, xy_drift[t]))

    # data formatting
    _data_out = da.stack(_translated_data)
    # just in case
    _data = da.swapaxes(_data, 0,1)
    _data_out = da.swapaxes(_data_out, 0,1)

    print('XY-shift has been scheduled.')
    
    return _data_out

def rotate_stack(_image, _alpha):

    # construct shifted stack
    _image_out = []
    
    for c,C in enumerate(_image):
        _out_stack = []
        for z,Z in enumerate(C):
            
            # here is the delayed function that allows to scatter everthing nicely across workers
            _out_stack.append(rotate(Z, -_alpha,reshape=False))
        
        # format arrays for proper export
        _out_stack = da.stack(_out_stack)
        _image_out.append(_out_stack)

    return da.stack(_image_out)
    


def apply_z_drift(_data, z_drift):

    # swap axes around
    _data = da.swapaxes(_data, 0,1)
    _data = da.swapaxes(_data, 3,2)

    _data_out = []
    for t,T in tqdm(enumerate(_data)):
        _data_out.append(translate_stack(T, z_drift[t]))

    # Stack the list to a dask array and swap back the axes.
    _data_out = da.stack(_data_out)
    _data = da.swapaxes(_data,2,3)
    _data = da.swapaxes(_data,0,1)
    _data_out = da.swapaxes(_data_out,2,3)
    _data_out = da.swapaxes(_data_out,0,1)

    return _data_out

def crop_data(data, xy_drift, z_drift):
    y_crop = [int(np.max(xy_drift[:,0])),int(np.shape(data)[-1]-abs(np.min(xy_drift[:,0])))]
    x_crop = [int(np.max(xy_drift[:,1])),int(np.shape(data)[-2]-abs(np.min(xy_drift[:,1])))]
    z_crop = [int(np.max(z_drift[:,0])),int(np.shape(data)[-3]-abs(np.min(z_drift[:,0])))]

    cropped = data[:,:,
                                                z_crop[0]:z_crop[1],
                                                y_crop[0]:y_crop[1],
                                                x_crop[0]:x_crop[1]]
    return cropped

def get_rotation(data, ref_channel, _registration_mode):
    
    if _registration_mode == 'mean':
        xy_movie = da.average(data[ref_channel], axis = 1).compute()
    elif _registration_mode == 'max':
        xy_movie = da.max(data[ref_channel], axis = 1).compute()
    elif _registration_mode == 'median': 
        xy_movie = da.median(data[ref_channel], axis = 1).compute()
        
    # get image radius
    radius = int(min(da.shape(data)[3], da.shape(data)[4])/2)

    print('Determining rotation')
    shifts, error, phasediff = [], [], []
    for t in tqdm(range(len(xy_movie)-1)):        

        s, e, p = phase_cross_correlation(  warp_polar(xy_movie[t], radius = radius),
                                            warp_polar(xy_movie[t+1], radius = radius), 
                                            normalization=None) 
        
        

        shifts.append(s)
        error.append(e)
        phasediff.append(p)
   
    shifts_a = np.cumsum(np.array(shifts), axis = 0)
    for i in shifts: 
        i[0] = 0
    
    shifts_a = shifts_a.tolist()
    shifts_a.insert(0,[0,0])
    shifts_a = np.asarray(shifts_a)
    
    plt.title('alpha-Drift')
    plt.plot(shifts_a[:,0], label = 'alpha')
    plt.legend()
    plt.xlabel('Timesteps')
    plt.ylabel('Rotation in degree')
    plt.savefig('Rotation-Drift.svg')
    plt.clf()

    return shifts_a[:,0]

def apply_alpha_drift(_data, alpha_drift):

    # swap axes around
    _data = da.swapaxes(_data, 0,1)

    _data_out = []
    for t,T in tqdm(enumerate(_data)):
        _data_out.append(rotate_stack(T, alpha_drift[t]))

    # Stack the list to a dask array and swap back the axes.
    _data_out = da.stack(_data_out)
    _data = da.swapaxes(_data,0,1)
    _data_out = da.swapaxes(_data_out,0,1)


    return _data_out

def write_tmp_data_to_disk(_path, _file, _current_file_path_index ,_new_shape='auto'): 
    print('Compute and save intermediate results to .zarr file on disk.')
    
    # check if current path is data or data2:
    # example file path: '.../tmp_data.zarr'
    
    if _current_file_path_index != 1: 
        _current_file_path_index = 1
        _path = _path.split('tmp_data.zarr')[0] + 'tmp_data_2.zarr'
    
    else: 
        _current_file_path_index = 2
        
    # writes with current shape and loads with a new one.
    with ProgressBar():
        da.to_zarr(_file,_path, overwrite = True)
    _file_reloaded = da.from_zarr(_path).rechunk(_new_shape)
    
    return _file_reloaded, _current_file_path_index


# Benchmarking notes: 
# File with 3.2Gb size, two channels, 21 timepoints,
# and 162 slices of 512, 512 pixels took 177.85 seconds (ca. 3 min)
# for computation using multiple saving steps. 