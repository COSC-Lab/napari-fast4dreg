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

import numpy as np
import dask.array as da
from pathlib import Path
import shutil
from typing import Union, Optional, Callable, Dict, Any
import warnings

from ._fast4Dreg_functions import (
    get_xy_drift,
    apply_xy_drift,
    get_z_drift,
    apply_z_drift,
    get_rotation,
    apply_alpha_drift,
    apply_beta_drift,
    apply_gamma_drift,
    crop_data,
    write_tmp_data_to_disk,
    read_tmp_data,
)


def register_image(
    image: Union[np.ndarray, da.Array],
    ref_channel: Union[int, str] = 0,
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
        Input image in CTZYX format (Channels, Time, Z, Y, X).
        Can be numpy array or dask array for out-of-memory processing.
    ref_channel : int or str, default=0
        Reference channel for drift detection. Can be:
        - Single channel: 0, 1, 2, etc.
        - Multiple channels (comma-separated): "0,3,5"
    output_dir : str or Path, default="./fast4dreg_output"
        Directory for output files and temporary storage.
    correct_xy : bool, default=True
        Apply XY (lateral) drift correction.
    correct_z : bool, default=True
        Apply Z (axial) drift correction.
    correct_rotation : bool, default=True
        Apply 3D rotation correction (XY, ZX, ZY planes).
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
        - 'registered_image': np.ndarray - The registered image (CTZYX)
        - 'xy_drift': np.ndarray - XY drift values (if return_drifts=True)
        - 'z_drift': np.ndarray - Z drift values (if return_drifts=True)
        - 'rotation_xy': np.ndarray - XY rotation angles (if return_drifts=True)
        - 'rotation_zx': np.ndarray - ZX rotation angles (if return_drifts=True)
        - 'rotation_zy': np.ndarray - ZY rotation angles (if return_drifts=True)
        - 'output_path': Path - Path to saved output
    
    Examples
    --------
    Basic registration with all corrections:
    
    >>> result = register_image(
    ...     image,
    ...     ref_channel=0,
    ...     output_dir="./results"
    ... )
    >>> registered = result['registered_image']
    
    Only XY correction with progress tracking:
    
    >>> def progress(msg):
    ...     print(f"Progress: {msg}")
    >>> 
    >>> result = register_image(
    ...     image,
    ...     ref_channel=1,
    ...     correct_xy=True,
    ...     correct_z=False,
    ...     correct_rotation=False,
    ...     progress_callback=progress
    ... )
    
    Multi-channel reference with normalization:
    
    >>> result = register_image(
    ...     image,
    ...     ref_channel="0,3,5",
    ...     normalize_channels=True,
    ...     projection_type='max'
    ... )
    
    Notes
    -----
    - Input should be in CTZYX format (Channels, Time, Z, Y, X)
    - Temporary Zarr stores are created in output_dir for out-of-memory processing
    - For large datasets, consider using dask arrays to avoid loading into RAM
    - The output is automatically saved to output_dir/registered.zarr
    
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
    
    # Validate shape early (before dask conversion)
    if isinstance(image, np.ndarray):
        if image.ndim != 5:
            raise ValueError(
                f"Image must be 5D (CTZYX format), got shape {image.shape}. "
                f"If your data has different dimensions, reshape it first."
            )
        _progress("Converting to dask array...")
        # Chunk to keep full channels and time together, chunk spatially
        chunks = (1, 1, 'auto', 'auto', 'auto')
        data = da.from_array(image, chunks=chunks)
    else:
        data = image
        # Validate dask array shape
        if data.ndim != 5:
            raise ValueError(
                f"Image must be 5D (CTZYX format), got shape {data.shape}. "
                f"If your data has different dimensions, reshape it first."
            )
    
    _progress(f"Input shape (CTZYX): {data.shape}")
    
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
    
    # Rotation correction
    if correct_rotation:
        _progress("Detecting 3D rotation...")
        alpha_xy, beta_zx, gamma_zy = get_rotation(
            data, ref_channel,
            projection_type=projection_type,
            reference_mode=reference_mode,
            normalize_channels=normalize_channels
        )
        drift_data['rotation_xy'] = alpha_xy
        drift_data['rotation_zx'] = beta_zx
        drift_data['rotation_zy'] = gamma_zy
        
        _progress("Applying XY rotation...")
        tmp_data = apply_alpha_drift(data, alpha_xy)
        tmp_data = write_tmp_data_to_disk(str(tmp_path_write), tmp_data, new_shape)
        tmp_path_read, tmp_path_write = tmp_path_write, tmp_path_read
        
        _progress("Applying ZX rotation...")
        tmp_data = apply_beta_drift(tmp_data, beta_zx)
        tmp_data = write_tmp_data_to_disk(str(tmp_path_write), tmp_data, new_shape)
        tmp_path_read, tmp_path_write = tmp_path_write, tmp_path_read
        
        _progress("Applying ZY rotation...")
        tmp_data = apply_gamma_drift(tmp_data, gamma_zy)
        tmp_data = write_tmp_data_to_disk(str(tmp_path_write), tmp_data, new_shape)
        tmp_path_read, tmp_path_write = tmp_path_write, tmp_path_read
        data = tmp_data
    
    # Crop if requested
    if crop_output and correct_xy and correct_z:
        _progress("Cropping output...")
        data = crop_data(data, drift_data.get('xy_drift'), drift_data.get('z_drift'))
        data = write_tmp_data_to_disk(str(tmp_path_write), data, new_shape)
        tmp_path_read, tmp_path_write = tmp_path_write, tmp_path_read
    
    # Compute final result
    _progress("Computing registered image...")
    registered_image = data.compute()
    
    # Save to Zarr
    _progress("Saving to Zarr...")
    zarr_path = output_dir / "registered.zarr"
    shape = registered_image.shape
    if len(shape) == 5:  # CTZYX
        chunks = (1, 1, shape[2], shape[3], shape[4])
    else:
        chunks = None
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
    axis_order: str = "CTZYX",
    **kwargs
) -> Dict[str, Any]:
    """
    Load and register an image from a file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to image file (supports TIFF, Zarr, NPY, etc.)
    axis_order : str, default="CTZYX"
        Axis order of the input file. Will be reordered to CTZYX.
        Common formats:
        - "CTZYX" - Already in correct format
        - "TZCYX" - ImageJ/Fiji format (Time, Z, Channels, Y, X)
        - "TZYX" - Single channel time series
    **kwargs
        Additional arguments passed to register_image()
    
    Returns
    -------
    dict
        Same as register_image()
    
    Examples
    --------
    >>> result = register_image_from_file(
    ...     "my_image.tif",
    ...     axis_order="TZCYX",  # ImageJ format
    ...     ref_channel=1
    ... )
    """
    import tifffile
    
    filepath = Path(filepath)
    
    # Load based on extension
    if filepath.suffix in ['.tif', '.tiff']:
        image = tifffile.imread(str(filepath))
    elif filepath.suffix == '.npy':
        image = np.load(str(filepath))
    elif filepath.suffix == '.zarr' or filepath.is_dir():
        import zarr
        image = da.from_zarr(str(filepath))
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    # Reorder axes to CTZYX
    if axis_order != "CTZYX":
        # Simple reordering logic
        if axis_order == "TZCYX":
            # Swap T and C, then T and Z
            image = np.swapaxes(image, 0, 2)  # CTZYX
            image = np.swapaxes(image, 1, 2)  # CTZYX
        elif axis_order == "TZYX":
            # Add channel dimension
            image = image[:, np.newaxis, ...]  # CTZYX
        else:
            warnings.warn(
                f"Axis order '{axis_order}' not recognized. "
                f"Please manually reorder to CTZYX format."
            )
    
    return register_image(image, **kwargs)


# Convenience aliases
fast4dreg = register_image  # Shorter alias
register = register_image   # Even shorter
