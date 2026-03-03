# Fast4Dreg.py package
# Intended Command Line usage: python Fast4Dreg.py image.tif output_dir
# dependencies:
#%%
# Imports
# Essentials
import numpy as np
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar
from skimage.transform import AffineTransform
from scipy.ndimage import affine_transform, rotate, shift
import zarr
import shutil
# Utility
from tqdm import tqdm
import sys
import os

# Optional GPU acceleration with pyclesperanto
try:
    import pyclesperanto as cle
    PYCLESPERANTO_AVAILABLE = True
except ImportError:
    PYCLESPERANTO_AVAILABLE = False
    cle = None

# Global flag to control GPU usage (can be set by user)
USE_GPU_ACCELERATION = False
GPU_INFO = "CPU (scipy)"

def get_gpu_info():
    """Get information about available GPU and current backend.
    
    Returns:
    --------
    str : Description of current processing backend (CPU or GPU with device name)
    """
    return GPU_INFO

def _detect_and_select_gpu():
    """Detect available GPUs and select the best one (prefer NVIDIA over Intel).
    
    Returns:
    --------
    tuple : (success: bool, device_name: str, device_info: str)
    """
    if not PYCLESPERANTO_AVAILABLE:
        return False, None, "CPU (scipy)"
    
    try:
        # Get all available devices
        available_devices = cle.available_device_names()
        print("Available OpenCL devices:")

        for i, device in enumerate(available_devices, 1):
            print(f"  {i}. {device}")

        if not available_devices:
            return False, None, "CPU (scipy) - No GPU devices found"
        
        # Prefer NVIDIA over Intel over others
        nvidia_devices = [d for d in available_devices if 'nvidia' in d.lower() or 'geforce' in d.lower() or 'rtx' in d.lower() or 'gtx' in d.lower()]
        intel_devices = [d for d in available_devices if 'intel' in d.lower()]
        
        selected_device = None
        if nvidia_devices:
            selected_device = nvidia_devices[0]
            print(f"Found NVIDIA GPU: {selected_device}")
        elif intel_devices:
            selected_device = intel_devices[0]
            print(f"Found Intel GPU: {selected_device}")
        else:
            selected_device = available_devices[0]
            print(f"Found GPU: {selected_device}")
        
        # Try to select the device
        cle.select_device(selected_device)
        device_info = f"GPU ({selected_device})"
        print(f"Successfully selected GPU: {selected_device}")
        return True, selected_device, device_info
        
    except Exception as e:
        print(f"Warning: Failed to initialize GPU: {e}")
        return False, None, "CPU (scipy) - GPU initialization failed"

def _is_gpu_oom_error(exc):
    message = str(exc).lower()
    return any(token in message for token in [
        "out of memory",
        "cl_out_of_resources",
        "cl_out_of_memory",
        "cl_mem_object_allocation_failure",
        "insufficient memory",
        "not enough memory",
    ])

def set_gpu_acceleration(enabled=True):
    """Enable or disable GPU acceleration using pyclesperanto.
    
    Parameters:
    -----------
    enabled : bool
        If True, use GPU acceleration (requires pyclesperanto_prototype).
        If False, use CPU (scipy) for transformations.
    
    Returns:
    --------
    bool : True if GPU acceleration was successfully enabled, False otherwise
    """
    global USE_GPU_ACCELERATION, GPU_INFO
    
    if enabled and not PYCLESPERANTO_AVAILABLE:
        print("Warning: pyclesperanto_prototype not available. Install with: pip install pyclesperanto-prototype")
        print("Falling back to CPU (scipy) processing.")
        USE_GPU_ACCELERATION = False
        GPU_INFO = "CPU (scipy)"
        return False
    
    if enabled:
        # Try to detect and select best GPU
        success, device_name, device_info = _detect_and_select_gpu()
        USE_GPU_ACCELERATION = success
        GPU_INFO = device_info
        if success:
            print(f"Transform backend set to: {device_info}")
        else:
            print(f"Transform backend set to: {device_info}")
        return success
    else:
        USE_GPU_ACCELERATION = False
        GPU_INFO = "CPU (scipy)"
        print(f"Transform backend set to: {GPU_INFO}")
        return False

# Automatically detect and enable GPU on module import
if PYCLESPERANTO_AVAILABLE:
    print("pyclesperanto_prototype available - detecting GPU...")
    _auto_success, _auto_device, _auto_info = _detect_and_select_gpu()
    if _auto_success:
        USE_GPU_ACCELERATION = True
        GPU_INFO = _auto_info
        print(f"✓ GPU acceleration enabled automatically: {_auto_info}")
    else:
        print(f"✗ GPU not available: {_auto_info}")
        GPU_INFO = _auto_info
else:
    print("pyclesperanto_prototype not installed - using CPU (scipy)")
    GPU_INFO = "CPU (scipy)"

def apply_projection(data, axis, projection_type='average'):
    """Apply projection along axis using specified method.
    
    Parameters:
    -----------
    data : dask array
        Input data
    axis : int
        Axis to project along
    projection_type : str
        One of 'average', 'max', 'median', 'min'
    """
    if projection_type == 'average':
        return da.average(data, axis=axis)
    elif projection_type == 'max':
        return da.max(data, axis=axis)
    elif projection_type == 'median':
        return da.median(data, axis=axis)
    elif projection_type == 'min':
        return da.min(data, axis=axis)
    else:
        raise ValueError(f"Unknown projection type: {projection_type}")


def get_reference_data(_data, ref_channel, normalize_channels=False):
    """Get reference channel data, supporting multiple channels.
    
    Parameters:
    -----------
    _data : dask array
        Input data with shape (C, T, Z, Y, X)
    ref_channel : int, list, or str
        Reference channel(s). Can be a single int, list of ints, 
        comma-separated string (e.g., "0,1,2"), or space-separated string (e.g., "0 1 2")
    normalize_channels : bool
        If True, normalize each channel before summing (when multiple channels)
    
    Returns:
    --------
    dask array with shape (T, Z, Y, X)
    """
    # Parse ref_channel
    if isinstance(ref_channel, str):
        # Handle comma-separated string like "0,3,5" or space-separated like "0 1 2"
        if ',' in ref_channel:
            ref_channels = [int(c.strip()) for c in ref_channel.split(',')]
        elif ' ' in ref_channel:
            ref_channels = [int(c.strip()) for c in ref_channel.split()]
        else:
            ref_channels = [int(ref_channel.strip())]
    elif isinstance(ref_channel, (list, tuple)):
        ref_channels = list(ref_channel)
    else:
        ref_channels = [int(ref_channel)]
    
    if len(ref_channels) == 1:
        # Single channel
        return _data[ref_channels[0]]
    else:
        # Multiple channels - sum them up
        print(f"Using multiple reference channels: {ref_channels}")
        
        if normalize_channels:
            print("Normalizing channels before summation")
            # Normalize each channel to 0-1 range before summing
            combined = None
            for idx, ch in enumerate(ref_channels):
                channel_data = _data[ch]
                # Normalize to 0-1
                ch_min = da.min(channel_data)
                ch_max = da.max(channel_data)
                normalized = (channel_data - ch_min) / (ch_max - ch_min + 1e-10)
                
                if combined is None:
                    combined = normalized
                else:
                    combined = combined + normalized
            return combined
        else:
            # Direct summation without normalization
            combined = None
            for ch in ref_channels:
                if combined is None:
                    combined = _data[ch]
                else:
                    combined = combined + _data[ch]
            return combined


def get_xy_drift(_data, ref_channel, projection_type='average', reference_mode='relative', normalize_channels=False):
    data_ref = get_reference_data(_data, ref_channel, normalize_channels)
    # Rechunk to ensure T dimension is fully loaded before averaging
    data_ref = data_ref.rechunk({0: -1, 1: 'auto', 2: 'auto', 3: 'auto'})
    xy_movie_avg = apply_projection(data_ref, axis=1, projection_type=projection_type)
    xy_movie = np.array(xy_movie_avg)

    # correct XY drift
    print(f'Determining drift in XY (projection: {projection_type}, reference: {reference_mode})')
    shifts, error, phasediff = [], [], []
    
    if reference_mode == 'relative':
        # Relative mode: compare each frame to the next
        for t in tqdm(range(len(xy_movie)-1)):        
            s, e, p = phase_cross_correlation(xy_movie[t],
                                                xy_movie[t+1], 
                                                normalization=None)
            shifts.append(s)
            error.append(e)
            phasediff.append(p)
        shifts_xy = np.cumsum(np.array(shifts), axis=0)
        shifts_xy = shifts_xy.tolist()
        shifts_xy.insert(0,[0,0])
    else:
        # First frame mode: compare all frames to the first frame
        for t in tqdm(range(1, len(xy_movie))):        
            s, e, p = phase_cross_correlation(xy_movie[0],
                                                xy_movie[t], 
                                                normalization=None)
            shifts.append(s)
            error.append(e)
            phasediff.append(p)
        shifts_xy = np.array(shifts).tolist()
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

def get_z_drift(_data, ref_channel, projection_type='average', reference_mode='relative', normalize_channels=False):
    # _data has shape (C, T, Z, Y, X)
    # _data[ref_channel] has shape (T, Z, Y, X)
    data_ref = get_reference_data(_data, ref_channel, normalize_channels)

    # Project over Y (axis 2) to get (T, Z, X) for ZX plane detection
    z_movie = apply_projection(data_ref, axis=2, projection_type=projection_type)
    z_movie = np.asarray(z_movie.compute())

    # correct Z drift
    print(f'Determining drift in Z (projection: {projection_type}, reference: {reference_mode})')
    shifts, error, phasediff = [], [], []
    
    if reference_mode == 'relative':
        # Relative mode: compare each frame to the next
        for t in tqdm(range(len(z_movie)-1)):        
            s, e, p = phase_cross_correlation(z_movie[t],
                                                z_movie[t+1], 
                                                normalization=None)
            shifts.append(s)
            error.append(e)
            phasediff.append(p)
        shifts_z = np.cumsum(np.array(shifts), axis=0)
        for i in shifts_z:
            i[1] = 0
        shifts_z = shifts_z.tolist()
        shifts_z.insert(0,[0,0])
    else:
        # First frame mode: compare all frames to the first frame
        for t in tqdm(range(1, len(z_movie))):        
            s, e, p = phase_cross_correlation(z_movie[0],
                                                z_movie[t], 
                                                normalization=None)
            shifts.append(s)
            error.append(e)
            phasediff.append(p)
        shifts_z = np.array(shifts)
        for i in shifts_z:
            i[1] = 0
        shifts_z = shifts_z.tolist()
        shifts_z.insert(0,[0,0])
    
    print(f"Shape of shifts: {np.asarray(shifts).shape}")
    print(f"Shifts: {shifts}")
    print(f"Errors: {error}")
    print(f"Phase differences: {phasediff}")
    shifts_z = np.asarray(shifts_z)

    plt.title('Z-Drift')
    plt.plot(shifts_z[:,0], label = 'z')
    plt.legend()
    plt.xlabel('Timesteps')
    plt.ylabel('Drift in pixel')
    plt.savefig('Z-Drift.svg')
    plt.clf()

    return shifts_z


def apply_xy_drift(_data, xy_drift):

    # Expectes _data to have the following shape:
    # np.shape_data = (channel, time, z, y, x)

    # Swap axes to iterate over time which aligns with chunks
    _data = da.swapaxes(_data, 0,1)
    
    print('Scheduling tasks...')
    _translated_data = []
    for t,T in tqdm(enumerate(_data)):
        _translated_data.append(translate_z_stack(T, xy_drift[t], transform_type='xy'))

    # data formatting
    _data_out = da.stack(_translated_data)
    # just in case
    _data = da.swapaxes(_data, 0,1)
    _data_out = da.swapaxes(_data_out, 0,1)

    print('XY-shift has been scheduled.')
    
    return _data_out
    


def translate_z_stack(_image, _shift, transform_type='z'):
    # _image has shape (channel, z, y, x)
    # _shift is the transformation parameter(s)
    # transform_type specifies which transformation to apply:
    #   'z': Z-direction translation [shift_z] - shifts in ZX plane
    #   'xy': XY plane translation [shift_y, shift_x] - applies uniformly to entire z-stack (3D volume)
    #   'alpha': XY plane rotation (around Z axis) - angle in degrees
    #   'beta': ZX plane rotation (around Y axis) - angle in degrees
    #   'gamma': ZY plane rotation (around X axis) - angle in degrees
    # Applies transformations to the entire z-stack at once (3D transformation, not plane-by-plane)
    
    # Choose backend based on global flag
    if USE_GPU_ACCELERATION and PYCLESPERANTO_AVAILABLE:
        return _translate_z_stack_gpu(_image, _shift, transform_type)
    else:
        return _translate_z_stack_cpu(_image, _shift, transform_type)


def _translate_z_stack_cpu(_image, _shift, transform_type):
    """CPU implementation using scipy.ndimage."""
    _image_out = []
    return_numpy = isinstance(_image, np.ndarray) and _image.ndim == 3
    image_iter = [_image] if return_numpy else _image

    for c, C in enumerate(image_iter):
        # C has shape (z, y, x) and is a dask array or numpy array
        use_numpy = return_numpy or isinstance(C, np.ndarray)

        if transform_type == 'z':
            # Z-direction translation: shift in z (dim 0) and x (dim 2), no shift in y (dim 1)
            shift_z = float(_shift[0])
            shift_x = float(_shift[1])
            offset = np.array([shift_z, 0.0, shift_x])
            if use_numpy:
                result = affine_transform(C, np.eye(3), offset=offset, order=1, mode='constant', cval=0.0)
            else:
                delayed_result = dask.delayed(affine_transform)(C, np.eye(3), offset=offset, order=1, mode='constant', cval=0.0)
                result = da.from_delayed(delayed_result, shape=C.shape, dtype=C.dtype)

        elif transform_type == 'xy':
            # XY plane translation: applied uniformly across entire z-stack (3D volume)
            offset = np.array([0.0, float(_shift[0]), float(_shift[1])])
            if use_numpy:
                result = affine_transform(C, np.eye(3), offset=offset, order=1, mode='constant', cval=0.0)
            else:
                delayed_result = dask.delayed(affine_transform)(C, np.eye(3), offset=offset, order=1, mode='constant', cval=0.0)
                result = da.from_delayed(delayed_result, shape=C.shape, dtype=C.dtype)

        elif transform_type == 'alpha':
            # XY plane rotation (rotation around Z axis) - applied uniformly to entire z-stack
            angle = float(_shift)
            if use_numpy:
                result = rotate(C, angle, axes=(1, 2), reshape=False, order=1, mode='constant', cval=0.0)
            else:
                delayed_result = dask.delayed(rotate)(C, angle, axes=(1, 2), reshape=False, order=1, mode='constant', cval=0.0)
                result = da.from_delayed(delayed_result, shape=C.shape, dtype=C.dtype)

        elif transform_type == 'beta':
            # ZX plane rotation (rotation around Y axis) - applied uniformly to entire z-stack
            angle = float(_shift)
            if use_numpy:
                result = rotate(C, angle, axes=(0, 2), reshape=False, order=1, mode='constant', cval=0.0)
            else:
                delayed_result = dask.delayed(rotate)(C, angle, axes=(0, 2), reshape=False, order=1, mode='constant', cval=0.0)
                result = da.from_delayed(delayed_result, shape=C.shape, dtype=C.dtype)

        elif transform_type == 'gamma':
            # ZY plane rotation (rotation around X axis) - applied uniformly to entire z-stack
            angle = float(_shift)
            if use_numpy:
                result = rotate(C, angle, axes=(0, 1), reshape=False, order=1, mode='constant', cval=0.0)
            else:
                delayed_result = dask.delayed(rotate)(C, angle, axes=(0, 1), reshape=False, order=1, mode='constant', cval=0.0)
                result = da.from_delayed(delayed_result, shape=C.shape, dtype=C.dtype)

        else:
            raise ValueError(f"Unknown transform_type: {transform_type}. Supported: 'z', 'xy', 'alpha', 'beta', 'gamma'")

        _image_out.append(result)

    return _image_out[0] if return_numpy else da.stack(_image_out)


def _translate_z_stack_gpu(_image, _shift, transform_type):
    """GPU implementation using pyclesperanto."""
    _image_out = []
    
    def _apply_gpu_transform(img_data, shift_params, ttype):
        """Helper to apply GPU transformation to numpy array."""
        global USE_GPU_ACCELERATION, GPU_INFO
        # Convert to numpy if dask array
        if hasattr(img_data, 'compute'):
            img_data = np.asarray(img_data.compute())
        else:
            img_data = np.asarray(img_data)
        try:
            # Push to GPU
            gpu_img = cle.push(img_data)

            if ttype == 'z':
                # Z-direction translation: shift in z (dim 0) and x (dim 2)
                shift_z = float(shift_params[0])
                shift_x = float(shift_params[1])
                # pyclesperanto uses (x, y, z) order, we have (z, y, x)
                # So translate with translateZ=shift_z, translateX=shift_x
                result_gpu = cle.rigid_transform(gpu_img, translate_x=shift_x, translate_y=0, translate_z=shift_z)

            elif ttype == 'xy':
                # XY plane translation across entire z-stack
                shift_y = float(shift_params[0])
                shift_x = float(shift_params[1])
                result_gpu = cle.rigid_transform(gpu_img, translate_x=shift_x, translate_y=shift_y, translate_z=0)

            elif ttype == 'alpha':
                # XY plane rotation (around Z axis)
                angle = float(shift_params)
                # Rotate in XY plane (around Z axis)
                result_gpu = cle.rigid_transform(gpu_img, angle_z=angle, centered=True)

            elif ttype == 'beta':
                # ZX plane rotation (around Y axis)
                angle = float(shift_params)
                result_gpu = cle.rigid_transform(gpu_img, angle_y=angle, centered=True)

            elif ttype == 'gamma':
                # ZY plane rotation (around X axis)
                angle = float(shift_params)
                result_gpu = cle.rigid_transform(gpu_img, angle_x=angle, centered=True)

            else:
                raise ValueError(f"Unknown transform_type: {ttype}")

            # Pull back from GPU and return
            result = cle.pull(result_gpu)
            return result.astype(img_data.dtype)
        except Exception as e:
            if _is_gpu_oom_error(e):
                print("Warning: GPU out of memory. Switching to CPU (scipy) for this transform and disabling GPU.")
                USE_GPU_ACCELERATION = False
                GPU_INFO = "CPU (scipy) - GPU OOM fallback"
                result = _translate_z_stack_cpu(img_data, shift_params, ttype)
                return result.astype(img_data.dtype)
            raise
    
    for c, C in enumerate(_image):
        # C has shape (z, y, x) and is a dask array
        delayed_result = dask.delayed(_apply_gpu_transform)(C, _shift, transform_type)
        _image_out.append(da.from_delayed(delayed_result, shape=C.shape, dtype=C.dtype))
    
    return da.stack(_image_out)


def apply_z_drift(_data, z_drift):
    
    # Swap axes to iterate over time which aligns with chunks
    _data = da.swapaxes(_data, 0,1)
    
    print('Scheduling tasks...')
    _translated_data = []
    for t,T in tqdm(enumerate(_data)):
        _translated_data.append(translate_z_stack(T, z_drift[t], transform_type='z'))

    # Stack the list to a dask array and swap back the axes.
    _data_out = da.stack(_translated_data)
    _data = da.swapaxes(_data, 0,1)
    _data_out = da.swapaxes(_data_out, 0,1)

    print('Z-shift has been scheduled.')
    
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

def get_rotation_alpha(data, ref_channel, projection_type='average', reference_mode='relative', normalize_channels=False):
    """Detect rotation in XY plane (alpha rotation)."""
    # Rechunk to ensure T dimension is fully loaded
    data_ref = get_reference_data(data, ref_channel, normalize_channels)
    data_ref = data_ref.rechunk({0: -1, 1: 'auto', 2: 'auto', 3: 'auto'})
    
    # Get rotation in XY plane (projecting over Z)
    xy_movie = np.array(apply_projection(data_ref, axis=1, projection_type=projection_type))
    radius_xy = int(min(da.shape(data)[3], da.shape(data)[4])/2)

    print(f'Determining rotation in XY plane (projection: {projection_type}, reference: {reference_mode})')
    
    # XY rotation detection
    shifts_xy = []
    if reference_mode == 'relative':
        for t in tqdm(range(len(xy_movie)-1), desc='XY rotation'):
            s, e, p = phase_cross_correlation(warp_polar(xy_movie[t], radius = radius_xy),
                                              warp_polar(xy_movie[t+1], radius = radius_xy), 
                                              normalization=None)
            shifts_xy.append(s)
        shifts_a_xy = np.cumsum(np.array(shifts_xy), axis = 0)
        shifts_a_xy = shifts_a_xy.tolist()
        shifts_a_xy.insert(0,[0,0])
    else:
        for t in tqdm(range(1, len(xy_movie)), desc='XY rotation'):
            s, e, p = phase_cross_correlation(warp_polar(xy_movie[0], radius = radius_xy),
                                              warp_polar(xy_movie[t], radius = radius_xy), 
                                              normalization=None)
            shifts_xy.append(s)
        shifts_a_xy = np.array(shifts_xy).tolist()
        shifts_a_xy.insert(0,[0,0])
    shifts_a_xy = np.asarray(shifts_a_xy)
    
    return shifts_a_xy[:,0]


def get_rotation_beta(data, ref_channel, projection_type='average', reference_mode='relative', normalize_channels=False):
    """Detect rotation in ZX plane (beta rotation)."""
    # Rechunk to ensure T dimension is fully loaded
    data_ref = get_reference_data(data, ref_channel, normalize_channels)
    data_ref = data_ref.rechunk({0: -1, 1: 'auto', 2: 'auto', 3: 'auto'})
    
    # Get rotation in ZX plane (projecting over Y)
    zx_movie = np.array(apply_projection(da.swapaxes(data_ref, 2, 1), axis=1, projection_type=projection_type))
    radius_zx = int(min(da.shape(data)[2], da.shape(data)[4])/2)

    print(f'Determining rotation in ZX plane (projection: {projection_type}, reference: {reference_mode})')
    
    # ZX rotation detection
    shifts_zx = []
    if reference_mode == 'relative':
        for t in tqdm(range(len(zx_movie)-1), desc='ZX rotation'):
            s, e, p = phase_cross_correlation(warp_polar(zx_movie[t], radius = radius_zx),
                                              warp_polar(zx_movie[t+1], radius = radius_zx),
                                              normalization=None)
            shifts_zx.append(s)
        shifts_a_zx = np.cumsum(np.array(shifts_zx), axis=0)
        shifts_a_zx = shifts_a_zx.tolist()
        shifts_a_zx.insert(0,[0,0])
    else:
        for t in tqdm(range(1, len(zx_movie)), desc='ZX rotation'):
            s, e, p = phase_cross_correlation(warp_polar(zx_movie[0], radius = radius_zx),
                                              warp_polar(zx_movie[t], radius = radius_zx),
                                              normalization=None)
            shifts_zx.append(s)
        shifts_a_zx = np.array(shifts_zx).tolist()
        shifts_a_zx.insert(0,[0,0])
    shifts_a_zx = np.asarray(shifts_a_zx)
    
    return shifts_a_zx[:,0]


def get_rotation_gamma(data, ref_channel, projection_type='average', reference_mode='relative', normalize_channels=False):
    """Detect rotation in ZY plane (gamma rotation)."""
    # Rechunk to ensure T dimension is fully loaded
    data_ref = get_reference_data(data, ref_channel, normalize_channels)
    data_ref = data_ref.rechunk({0: -1, 1: 'auto', 2: 'auto', 3: 'auto'})
    
    # Get rotation in ZY plane (projecting over X)
    zy_movie = np.array(apply_projection(da.swapaxes(data_ref, 3, 1), axis=1, projection_type=projection_type))
    radius_zy = int(min(da.shape(data)[2], da.shape(data)[3])/2)

    print(f'Determining rotation in ZY plane (projection: {projection_type}, reference: {reference_mode})')
    
    # ZY rotation detection
    shifts_zy = []
    if reference_mode == 'relative':
        for t in tqdm(range(len(zy_movie)-1), desc='ZY rotation'):
            s, e, p = phase_cross_correlation(warp_polar(zy_movie[t], radius = radius_zy),
                                              warp_polar(zy_movie[t+1], radius = radius_zy),
                                              normalization=None)
            shifts_zy.append(s)
        shifts_a_zy = np.cumsum(np.array(shifts_zy), axis=0)
        shifts_a_zy = shifts_a_zy.tolist()
        shifts_a_zy.insert(0,[0,0])
    else:
        for t in tqdm(range(1, len(zy_movie)), desc='ZY rotation'):
            s, e, p = phase_cross_correlation(warp_polar(zy_movie[0], radius = radius_zy),
                                              warp_polar(zy_movie[t], radius = radius_zy),
                                              normalization=None)
            shifts_zy.append(s)
        shifts_a_zy = np.array(shifts_zy).tolist()
        shifts_a_zy.insert(0,[0,0])
    shifts_a_zy = np.asarray(shifts_a_zy)
    
    return shifts_a_zy[:,0]


def get_rotation(data, ref_channel, projection_type='average', reference_mode='relative', normalize_channels=False):
    """Detect rotation in all three planes (XY, ZX, ZY) - legacy function that detects all at once."""
    # Get all rotations
    alpha_xy = get_rotation_alpha(data, ref_channel, projection_type, reference_mode, normalize_channels)
    beta_zx = get_rotation_beta(data, ref_channel, projection_type, reference_mode, normalize_channels)
    gamma_zy = get_rotation_gamma(data, ref_channel, projection_type, reference_mode, normalize_channels)
    
    # Plot results for all three planes
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].set_title('XY-Rotation (alpha)')
    axes[0].plot(alpha_xy, label = 'alpha')
    axes[0].legend()
    axes[0].set_xlabel('Timesteps')
    axes[0].set_ylabel('Rotation in degree')
    
    axes[1].set_title('ZX-Rotation (beta)')
    axes[1].plot(beta_zx, label = 'beta')
    axes[1].legend()
    axes[1].set_xlabel('Timesteps')
    axes[1].set_ylabel('Rotation in degree')
    
    axes[2].set_title('ZY-Rotation (gamma)')
    axes[2].plot(gamma_zy, label = 'gamma')
    axes[2].legend()
    axes[2].set_xlabel('Timesteps')
    axes[2].set_ylabel('Rotation in degree')
    
    plt.tight_layout()
    plt.savefig('Rotation-Drift.svg')
    plt.clf()

    return alpha_xy, beta_zx, gamma_zy


def apply_alpha_drift(_data, alpha_xy):
    # Applies XY plane rotation (around Z axis) to entire z-stacks
    # alpha_xy is array of rotation angles for each timepoint
    
    # Swap axes to iterate over time which aligns with chunks
    _data = da.swapaxes(_data, 0,1)

    _data_out = []
    for t, T in tqdm(enumerate(_data), desc='Applying XY rotation'):
        result = translate_z_stack(T, alpha_xy[t], transform_type='alpha')
        _data_out.append(result)

    # Stack the list to a dask array and swap back the axes.
    _data_out = da.stack(_data_out)
    _data = da.swapaxes(_data, 0,1)
    _data_out = da.swapaxes(_data_out, 0,1)

    print('XY rotation (alpha) has been scheduled.')
    
    return _data_out


def apply_beta_drift(_data, beta_zx):
    # Applies ZX plane rotation (around Y axis) to entire z-stacks
    # beta_zx is array of rotation angles for each timepoint
    
    # Swap axes to iterate over time which aligns with chunks
    _data = da.swapaxes(_data, 0,1)

    _data_out = []
    for t, T in tqdm(enumerate(_data), desc='Applying ZX rotation'):
        result = translate_z_stack(T, beta_zx[t], transform_type='beta')
        _data_out.append(result)

    # Stack the list to a dask array and swap back the axes.
    _data_out = da.stack(_data_out)
    _data = da.swapaxes(_data, 0,1)
    _data_out = da.swapaxes(_data_out, 0,1)

    print('ZX rotation (beta) has been scheduled.')
    
    return _data_out


def apply_gamma_drift(_data, gamma_zy):
    # Applies ZY plane rotation (around X axis) to entire z-stacks
    # gamma_zy is array of rotation angles for each timepoint
    
    # Swap axes to iterate over time which aligns with chunks
    _data = da.swapaxes(_data, 0,1)

    _data_out = []
    for t, T in tqdm(enumerate(_data), desc='Applying ZY rotation'):
        result = translate_z_stack(T, gamma_zy[t], transform_type='gamma')
        _data_out.append(result)

    # Stack the list to a dask array and swap back the axes.
    _data_out = da.stack(_data_out)
    _data = da.swapaxes(_data, 0,1)
    _data_out = da.swapaxes(_data_out, 0,1)

    print('ZY rotation (gamma) has been scheduled.')
    
    return _data_out

def write_tmp_data_to_disk(_path, _file, _new_shape=None): 
    """Write temporary data to Zarr format with optimized chunking.
    
    Uses (1, 1, Z, Y, X) chunking for efficient per-channel access.
    This is especially beneficial for multi-channel images.
    
    Parameters:
    -----------
    _path : str
        Path to store the Zarr array
    _file : dask array
        Data to save, expected shape (C, T, Z, Y, X)
    _new_shape : tuple, optional
        Desired chunk shape. If None, uses (1, 1, full_Z, full_Y, full_X)
    
    Returns:
    --------
    dask array backed by Zarr store
    """
    print(f'Saving intermediate results to Zarr format at {_path}')
    
    # Determine optimal chunking: (C=1, T=1, Z=full, Y=full, X=full)
    if _new_shape is None:
        shape = _file.shape
        # For CTZYX format: chunk by (1, 1, full_Z, full_Y, full_X)
        _new_shape = (1, 1, shape[2], shape[3], shape[4])
    
    # Rechunk before saving for optimal write performance
    _file_rechunked = _file.rechunk(_new_shape)
    
    # Save to Zarr with progress bar
    with ProgressBar():
        _file_rechunked.to_zarr(_path, overwrite=True)
    
    # Reload as dask array backed by Zarr
    zarr_array = zarr.open(_path, mode='r+')
    _file_reloaded = da.from_zarr(zarr_array).rechunk(_new_shape)
    
    print(f'  Data shape: {_file_reloaded.shape}, chunks: {_file_reloaded.chunks}')
    return _file_reloaded


def read_tmp_data(_path, _new_shape=None): 
    """Read temporary data from Zarr format.
    
    Parameters:
    -----------
    _path : str
        Path to the Zarr array
    _new_shape : tuple, optional
        Desired chunk shape. If None, keeps existing chunks
    
    Returns:
    --------
    dask array backed by Zarr store
    """
    print(f'Loading data from Zarr format at {_path}')
    zarr_array = zarr.open(_path, mode='r+')
    _file_loaded = da.from_zarr(zarr_array)
    
    if _new_shape is not None:
        _file_loaded = _file_loaded.rechunk(_new_shape)
    
    print(f'  Data shape: {_file_loaded.shape}, chunks: {_file_loaded.chunks}')
    return _file_loaded

# Storage format notes:
# - Uses Zarr format instead of NPY stacks for better performance
# - Chunking strategy: (C=1, T=1, Z=full, Y=full, X=full)
# - This allows independent, efficient access to individual channels and timepoints
# - Built-in compression (Blosc/LZ4) reduces disk usage by ~2-3x
# - Better random access patterns compared to NPY stacks
# - Particularly efficient for multi-channel images 
