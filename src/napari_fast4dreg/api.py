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
        Input image in specified axis order (default: CTZYX).
        Can be numpy array or dask array for out-of-memory processing.
    axis_order : str, default="CTZYX"
        Axis order specification as a string. Examples:
        - "CTZYX": Channels, Time, Z, Y, X (standard 5D)
        - "TZYX": Time, Z, Y, X (4D, single channel)
        - "ZYX": Z, Y, X (3D, single timepoint single channel)
        - "CYX": Channels, Y, X (2D)
        Supports any combination of C, T, Z, Y, X where Y and X are required.
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
        - 'registered_image': np.ndarray - The registered image (CTZYX)
        - 'xy_drift': np.ndarray - XY drift values (if return_drifts=True)
        - 'z_drift': np.ndarray - Z drift values (if return_drifts=True)
        - 'rotation_xy': np.ndarray - XY rotation angles (if return_drifts=True)
        - 'rotation_zx': np.ndarray - ZX rotation angles (if return_drifts=True)
        - 'rotation_zy': np.ndarray - ZY rotation angles (if return_drifts=True)
        - 'output_path': Path - Path to saved output
    
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
    ...     ref_channel="0,3,5",
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

    # Compute final result
    _progress("Computing registered image...")
    registered_image = data.compute()

    # Save to Zarr
    _progress("Saving to Zarr...")
    zarr_path = output_dir / "registered.zarr"
    
    # Revert to original axis order
    _progress(f"Reverting to original axis order ({axis_order})...")
    registered_image = revert_to_original_axis_order(registered_image, axis_order)
    
    shape = registered_image.shape
    if len(shape) == 5:  # CTZYX
        chunks = (1, 1, shape[2], shape[3], shape[4])
    elif len(shape) == 4:
        chunks = (1, shape[1], shape[2], shape[3])
    elif len(shape) == 3:
        chunks = shape  # Use full shape for 3D arrays
    else:
        chunks = shape  # Use full shape for 2D or other cases
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
        image = da.from_zarr(str(filepath))
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    # Reorder axes to CTZYX
    if axis_order != "CTZYX":
        # Simple reordering logic
        if axis_order == "TZCYX":
            # Swap T and C, then T and Z
            # Use image-specific method (works for both numpy and dask)
            if isinstance(image, da.Array):
                image = da.swapaxes(image, 0, 2)  # CTZYX
                image = da.swapaxes(image, 1, 2)  # CTZYX
            else:
                image = np.swapaxes(image, 0, 2)  # CTZYX
                image = np.swapaxes(image, 1, 2)  # CTZYX
        elif axis_order == "TZYX":
            # Add channel dimension at position 0
            if isinstance(image, da.Array):
                image = image[da.newaxis, :, ...]  # CTZYX
            else:
                image = image[np.newaxis, :, ...]  # CTZYX
        else:
            warnings.warn(
                f"Axis order '{axis_order}' not recognized. "
                f"Please manually reorder to CTZYX format."
            )

    return register_image(image, **kwargs)


# Convenience aliases
fast4dreg = register_image  # Shorter alias
register = register_image   # Even shorter
