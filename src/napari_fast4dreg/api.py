"""
Programmatic API for napari-fast4dreg.

This module provides a clean, high-level API for integrating
Fast4DReg registration into workflows without the napari GUI.

Example usage:
    >>> import numpy as np
    >>> from napari_fast4dreg import register_image
    >>> 
    >>> # Load your image (CTZYX format)
    >>> image = np.load("my_image.npy")
    >>> 
    >>> # Run registration
    >>> result = register_image(
    ...     image,
    ...     ref_channel=0,
    ...     correct_xy=True,
    ...     correct_z=True,
    ...     correct_rotation=True,
    ...     output_dir="./results"
    ... )
    >>> 
    >>> # Access results
    >>> registered = result['registered_image']
    >>> xy_drift = result['xy_drift']
    >>> z_drift = result['z_drift']
"""

import shutil
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import dask.array as da
import numpy as np

from ._fast4Dreg_functions import (
    apply_alpha_drift,
    apply_beta_drift,
    apply_gamma_drift,
    apply_xy_drift,
    apply_z_drift,
    crop_data,
    get_rotation_alpha,
    get_rotation_beta,
    get_rotation_gamma,
    get_xy_drift,
    get_z_drift,
    write_tmp_data_to_disk,
)
from ._axis_utils import convert_to_ctzyx, revert_to_original_axis_order


def register_image(
    image: Union[np.ndarray, da.Array],
    axis_order: str = "CTZYX",
    ref_channel: Union[int, str, list, tuple] = 0,
    output_dir: Union[str, Path] = "./fast4dreg_output",
    correct_xy: bool = True,
    correct_z: bool = True,
    correct_rotation: bool = True,
    crop_output: bool = False,
    projection_type: str = 'average',
    reference_mode: str = 'relative',
    normalize_channels: bool = False,
    keep_temp_files: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    return_drifts: bool = True,
) -> Dict[str, Any]:
    """
    Register a multi-dimensional image using Fast4DReg algorithm.
    
    This function performs 4D drift correction on time-lapse volumetric images,
    applying XY, Z, and 3D rotation corrections as specified.
    
    Parameters
    ----------
    image : np.ndarray or dask.array.Array
        Input image in specified axis order (default: CTZYX).
        Can be numpy array or dask array for out-of-memory processing.
    axis_order : str, default="CTZYX"
        Axis order specification as a string. Examples:
        - "CTZYX": Channels, Time, Z, Y, X (standard 5D)
        - "TZYX": Time, Z, Y, X (4D, single channel)
        - "ZYX": Z, Y, X (3D, single timepoint single channel)
        - "CYX": Channels, Y, X (2D)
        Supports any combination of C, T, Z, Y, X where Y and X are required.
    ref_channel : int, str, list, or tuple, default=0
        Reference channel(s) for drift detection. Can be:
        - Single channel: 0, 1, 2 (int)
        - List of channels: [0, 3, 5] or (0, 3, 5)
        - Comma-separated string: "0,3,5"
        - Space-separated string: "0 3 5"
        When multiple channels are specified, they are summed together
        (optionally normalized if normalize_channels=True)
    output_dir : str or Path, default="./fast4dreg_output"
        Directory for output files and temporary storage.
    correct_xy : bool, default=True
        Apply XY (lateral) drift correction.
    correct_z : bool, default=True
        Apply Z (axial) drift correction.
    correct_rotation : bool, default=True
        Apply 3D rotation correction sequentially (XY, then ZX, then ZY planes).
        Each rotation is detected and applied on the already-corrected data from
        the previous step for improved accuracy.
    crop_output : bool, default=False
        Crop output to remove invalid regions after correction.
    projection_type : str, default='average'
        Projection method for Z-stacks: 'average', 'max', 'median', or 'min'.
    reference_mode : str, default='relative'
        Drift detection mode:
        - 'relative': Frame-to-frame comparison (cumulative)
        - 'first_frame': Compare all frames to first frame
    normalize_channels : bool, default=False
        Normalize multiple reference channels before summing.
    keep_temp_files : bool, default=False
        Keep temporary Zarr stores after completion.
    progress_callback : callable, optional
        Function to call with progress updates: callback(message: str)
    return_drifts : bool, default=True
        Include drift arrays in the returned dictionary.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'registered_image': dask.array.Array - The registered image in the same 
          axis order as specified by the `axis_order` parameter. Lazy-loaded from Zarr.
        - 'xy_drift': np.ndarray - XY drift values (if return_drifts=True)
        - 'z_drift': np.ndarray - Z drift values (if return_drifts=True)
        - 'rotation_xy': np.ndarray - XY rotation angles (if return_drifts=True)
        - 'rotation_zx': np.ndarray - ZX rotation angles (if return_drifts=True)
        - 'rotation_zy': np.ndarray - ZY rotation angles (if return_drifts=True)
        - 'output_path': Path - Path to saved output (Zarr format)
    
    Examples
    --------
    Basic registration with CTZYX format:
    
    >>> result = register_image(
    ...     image,  # shape: (2, 10, 50, 512, 512) - CTZYX
    ...     axis_order="CTZYX",
    ...     ref_channel=0,
    ...     output_dir="./results"
    ... )
    >>> registered = result['registered_image']
    
    TZYX format (4D, single channel):
    
    >>> result = register_image(
    ...     image,  # shape: (10, 50, 512, 512) - TZYX
    ...     axis_order="TZYX",
    ...     ref_channel=0,
    ...     output_dir="./results"
    ... )
    
    ZYX format (3D, single timepoint):
    
    >>> result = register_image(
    ...     image,  # shape: (50, 512, 512) - ZYX
    ...     axis_order="ZYX",
    ...     output_dir="./results"
    ... )
    
    Only XY correction with progress tracking:
    
    >>> def progress(msg):
    ...     print(f"Progress: {msg}")
    >>> 
    >>> result = register_image(
    ...     image,
    ...     axis_order="TZYX",
    ...     ref_channel=1,
    ...     correct_xy=True,
    ...     correct_z=False,
    ...     correct_rotation=False,
    ...     progress_callback=progress
    ... )
    
    Multi-channel reference with normalization:
    
    >>> result = register_image(
    ...     image,
    ...     axis_order="CTZYX",
    ...     ref_channel=[0, 3, 5],  # Can also use "0,3,5" or (0, 3, 5)
    ...     normalize_channels=True,
    ...     projection_type='max'
    ... )
    
    Notes
    -----
    - Input axis order is automatically converted to CTZYX internally for processing
    - Output is returned in the same axis order as the input
    - Temporary Zarr stores are created in output_dir for out-of-memory processing
    - For large datasets, consider using dask arrays to avoid loading into RAM
    - The output is automatically saved to output_dir/registered.zarr
    - Missing dimensions (C, T, Z) are added automatically as singletons
    
    See Also
    --------
    register_image_from_file : Load and register image from file
    """

    def _progress(message: str):
        """Internal progress callback wrapper."""
        if progress_callback is not None:
            progress_callback(message)

    # Setup paths
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to dask array early
    _progress(f"Converting image to dask array (axis order: {axis_order})...")
    img = da.asarray(image)
    original_shape = img.shape
    
    # Validate and convert to CTZYX format
    _progress(f"Converting from {axis_order} to CTZYX format...")
    data, single_channel_mode, original_ndim = convert_to_ctzyx(img, axis_order)
    
    _progress(f"Input shape ({axis_order}): {original_shape}")
    _progress(f"Working shape (CTZYX): {data.shape}")

    # Setup temporary storage
    tmp_path_1 = output_dir / "tmp_data_1.zarr"
    tmp_path_2 = output_dir / "tmp_data_2.zarr"
    tmp_path_read = tmp_path_1
    tmp_path_write = tmp_path_2

    # Clean up old temp files
    for tmp_path in [tmp_path_1, tmp_path_2]:
        if tmp_path.exists():
            shutil.rmtree(tmp_path)

    # Track drift values
    drift_data = {}

    # Get chunking
    new_shape = data.chunksize

    # Write initial data
    _progress("Writing to temporary storage...")
    data = write_tmp_data_to_disk(str(tmp_path_write), data, new_shape)
    tmp_path_read, tmp_path_write = tmp_path_write, tmp_path_read

    # XY correction
    if correct_xy:
        _progress("Detecting XY drift...")
        xy_drift = get_xy_drift(
            data, ref_channel,
            projection_type=projection_type,
            reference_mode=reference_mode,
            normalize_channels=normalize_channels
        )
        drift_data['xy_drift'] = xy_drift

        _progress("Applying XY correction...")
        tmp_data = apply_xy_drift(data, xy_drift)
        tmp_data = write_tmp_data_to_disk(str(tmp_path_write), tmp_data, new_shape)
        tmp_path_read, tmp_path_write = tmp_path_write, tmp_path_read
        data = tmp_data

    # Z correction
    if correct_z:
        _progress("Detecting Z drift...")
        z_drift = get_z_drift(
            data, ref_channel,
            projection_type=projection_type,
            reference_mode=reference_mode,
            normalize_channels=normalize_channels
        )
        drift_data['z_drift'] = z_drift

        _progress("Applying Z correction...")
        tmp_data = apply_z_drift(data, z_drift)
        tmp_data = write_tmp_data_to_disk(str(tmp_path_write), tmp_data, new_shape)
        tmp_path_read, tmp_path_write = tmp_path_write, tmp_path_read
        data = tmp_data

    # Rotation correction - Sequential estimation and application
    if correct_rotation:
        # Alpha (XY plane) rotation
        _progress("Detecting XY plane rotation (alpha)...")
        alpha_xy = get_rotation_alpha(
            data, ref_channel,
            projection_type=projection_type,
            reference_mode=reference_mode,
            normalize_channels=normalize_channels
        )
        drift_data['rotation_xy'] = alpha_xy

        _progress("Applying XY rotation (alpha)...")
        tmp_data = apply_alpha_drift(data, alpha_xy)
        tmp_data = write_tmp_data_to_disk(str(tmp_path_write), tmp_data, new_shape)
        tmp_path_read, tmp_path_write = tmp_path_write, tmp_path_read
        data = tmp_data

        # Beta (ZX plane) rotation
        _progress("Detecting ZX plane rotation (beta)...")
        beta_zx = get_rotation_beta(
            data, ref_channel,
            projection_type=projection_type,
            reference_mode=reference_mode,
            normalize_channels=normalize_channels
        )
        drift_data['rotation_zx'] = beta_zx

        _progress("Applying ZX rotation (beta)...")
        tmp_data = apply_beta_drift(data, beta_zx)
        tmp_data = write_tmp_data_to_disk(str(tmp_path_write), tmp_data, new_shape)
        tmp_path_read, tmp_path_write = tmp_path_write, tmp_path_read
        data = tmp_data

        # Gamma (ZY plane) rotation
        _progress("Detecting ZY plane rotation (gamma)...")
        gamma_zy = get_rotation_gamma(
            data, ref_channel,
            projection_type=projection_type,
            reference_mode=reference_mode,
            normalize_channels=normalize_channels
        )
        drift_data['rotation_zy'] = gamma_zy

        _progress("Applying ZY rotation (gamma)...")
        tmp_data = apply_gamma_drift(data, gamma_zy)
        tmp_data = write_tmp_data_to_disk(str(tmp_path_write), tmp_data, new_shape)
        tmp_path_read, tmp_path_write = tmp_path_write, tmp_path_read
        data = tmp_data

    # Crop if requested
    if crop_output and correct_xy and correct_z:
        _progress("Cropping output...")
        data = crop_data(data, drift_data.get('xy_drift'), drift_data.get('z_drift'))
        data = write_tmp_data_to_disk(str(tmp_path_write), data, new_shape)
        tmp_path_read, tmp_path_write = tmp_path_write, tmp_path_read

    # Revert to original axis order
    _progress(f"Reverting to original axis order ({axis_order})...")
    registered_image = revert_to_original_axis_order(data, axis_order)
    
    shape = registered_image.shape
    if len(shape) == 5:  # CTZYX
        chunks = (1, 1, shape[2], shape[3], shape[4])
    elif len(shape) == 4:
        chunks = (1, shape[1], shape[2], shape[3])
    elif len(shape) == 3:
        chunks = shape  # Use full shape for 3D arrays
    else:
        chunks = shape  # Use full shape for 2D or other cases
    
    # Save to Zarr
    _progress("Saving to Zarr...")
    zarr_path = output_dir / "registered.zarr"
    # Use rechunk if already a dask array, otherwise from_array
    if isinstance(registered_image, da.Array):
        da.rechunk(registered_image, chunks=chunks).to_zarr(str(zarr_path), overwrite=True)
    else:
        da.from_array(registered_image, chunks=chunks).to_zarr(str(zarr_path), overwrite=True)

    # Clean up temp files
    if not keep_temp_files:
        _progress("Cleaning up temporary files...")
        for tmp_path in [tmp_path_1, tmp_path_2]:
            if tmp_path.exists():
                shutil.rmtree(tmp_path)

    _progress("Registration complete!")

    # Build result dictionary
    result = {
        'registered_image': registered_image,
        'output_path': zarr_path,
    }

    if return_drifts:
        result.update(drift_data)

    return result


def register_image_from_file(
    filepath: Union[str, Path],
    axis_order: str = "TZCYX",
    **kwargs
) -> Dict[str, Any]:
    """
    Load and register an image from a file.
    
    This function loads an image file in any axis order and registers it.
    The output is returned in the same axis order as specified.
    
    Parameters
    ----------
    filepath : str or Path
        Path to image file. Supported formats:
        - TIFF (.tif, .tiff) - Most common for microscopy data
        - NumPy binary (.npy) - For numpy arrays
        - Zarr (.zarr) - For chunked out-of-memory datasets
    axis_order : str, default="TZCYX"
        Axis order of the input file. Supports any combination of C, T, Z, Y, X
        where Y and X are required. Examples:
        - "TZCYX" - ImageJ/Fiji format (Time, Z, Channels, Y, X) [DEFAULT]
        - "CTZYX" - Standard 5D format
        - "TZYX" - Single channel time series
        - "ZYX" - Single timepoint, single channel volume
        - "CZYX" - Multi-channel, single timepoint
        - "YX" - 2D image
        Missing dimensions (C, T, Z) are added as singletons automatically.
    **kwargs
        Additional arguments passed to register_image() (ref_channel, output_dir, 
        correct_xy, correct_z, correct_rotation, etc.)
    
    Returns
    -------
    dict
        Dictionary with same structure as register_image(). The 'registered_image' 
        is returned in the same axis order as the input:
        - dask.array.Array - Lazy-loaded from Zarr
        - Format matches the input axis_order parameter
        
        All other return values (drift data, output_path) are the same as register_image().
    
    Examples
    --------
    ImageJ/Fiji TIFF format (input TZCYX → output TZCYX):
    
    >>> result = register_image_from_file(
    ...     "my_image.tif",
    ...     axis_order="TZCYX",  # ImageJ format
    ...     ref_channel=1,
    ...     output_dir="./results"
    ... )
    >>> registered = result['registered_image']  # Shape: (T, Z, C, Y, X)
    >>> xy_drift = result['xy_drift']  # Shape: (T, 2)
    
    Single channel time series (input TZYX → output TZYX):
    
    >>> result = register_image_from_file(
    ...     "timelapse.tif",
    ...     axis_order="TZYX",
    ...     correct_xy=True,
    ...     correct_z=True
    ... )
    >>> registered = result['registered_image']  # Shape: (T, Z, Y, X)
    
    3D volume, single timepoint (input ZYX → output ZYX):
    
    >>> result = register_image_from_file(
    ...     "volume.zarr",
    ...     axis_order="ZYX",
    ...     correct_rotation=True
    ... )
    >>> registered = result['registered_image']  # Shape: (Z, Y, X)
    
    Multi-channel Z-stack (input CZYX → output CZYX):
    
    >>> result = register_image_from_file(
    ...     "stack.npy",
    ...     axis_order="CZYX",
    ...     ref_channel="0,1"
    ... )
    >>> registered = result['registered_image']  # Shape: (C, Z, Y, X)
    
    Notes
    -----
    - This function loads an image and passes it to register_image() with the specified axis_order
    - Output preserves the input axis order format
    - Missing dimensions (C, T, Z) are added as singletons during processing
    - The registered image is stored in Zarr format for out-of-memory access
    
    See Also
    --------
    register_image : For custom axis order in output, use this function directly
    """
    import tifffile

    filepath = Path(filepath)

    # Load based on extension
    if filepath.suffix in ['.tif', '.tiff']:
        image = da.from_zarr(tifffile.imread(str(filepath), aszarr=True))
    
    # npy support for test files 
    elif filepath.suffix == '.npy': 
        image = np.load(str(filepath))
    
    elif filepath.suffix == '.zarr' or filepath.is_dir():
        image = da.from_zarr(str(filepath))
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    # Call register_image with CTZYX format to ensure output is always CTZYX
    return register_image(image, axis_order=axis_order, **kwargs)


# Convenience aliases
fast4dreg = register_image  # Shorter alias
register = register_image   # Even shorter
