#!/usr/bin/env python
"""
Test script for napari-fast4dreg registration pipeline.
Loads example data and runs the registration without the napari UI.
"""

import os
import sys
from pathlib import Path

import dask.array as da
import tifffile

# Add src to path to import directly from module (avoid Qt dependencies)
sys.path.insert(0, str(Path(__file__).parent / "src" / "napari_fast4dreg"))

# Import only the functions module to avoid Qt/napari widget imports
from _fast4Dreg_functions import (
    apply_alpha_drift,
    apply_beta_drift,
    apply_gamma_drift,
    apply_xy_drift,
    apply_z_drift,
    get_rotation_alpha,
    get_rotation_beta,
    get_rotation_gamma,
    get_xy_drift,
    get_z_drift,
    write_tmp_data_to_disk,
)


def main():
    # Setup paths
    example_dir = Path(__file__).parent / "example_files"
    output_dir = example_dir
    tif_file = list(example_dir.glob("*.tif"))[0]

    print(f"Loading test image: {tif_file}")

    # Load test image (TZCYX format from ImageJ)
    image = tifffile.imread(tif_file)
    print(f"Loaded image shape: {image.shape}")

    # Convert to dask array with explicit chunks to avoid 'auto' chunking issues
    # Keep full T in one chunk, chunk other dims
    image_dask = da.from_array(image, chunks=(21, -1, -1, -1, -1))
    # Swap axes to CTZYX format
    image_dask = image_dask.swapaxes(0, 2)  # CTZYX
    image_dask = image_dask.swapaxes(1, 2)  # CTZYX
    # Rechunk to ensure all T is loaded together per channel
    image_dask = image_dask.rechunk({0: -1, 1: -1, 2: 'auto', 3: 'auto', 4: 'auto'})

    print(f"Reshaped to CTZYX: {image_dask.shape}")

    data = image_dask
    new_shape = data.chunksize
    ref_channel = 0

    os.chdir(output_dir)
    # Use two temporary directories and alternate between them
    tmp_path_1 = str(output_dir / "tmp_data_1")
    tmp_path_2 = str(output_dir / "tmp_data_2")
    tmp_path_read = tmp_path_1
    tmp_path_write = tmp_path_2

    # Clean up old tmp_data directories
    import shutil
    for tmp_path in [tmp_path_1, tmp_path_2]:
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)

    print(f"Output directory: {output_dir}")

    # Step 1: Write to disk
    print("\n[1/5] Writing data to temporary storage...")
    data = write_tmp_data_to_disk(tmp_path_write, data, new_shape)
    # Swap paths for next iteration
    tmp_path_read, tmp_path_write = tmp_path_write, tmp_path_read

    # Step 2: XY correction
    print("\n[2/5] Detecting XY drift...")
    xy_drift = get_xy_drift(data, ref_channel)
    tmp_data = apply_xy_drift(data, xy_drift)
    tmp_data = write_tmp_data_to_disk(tmp_path_write, tmp_data, new_shape)
    # Swap paths for next iteration
    tmp_path_read, tmp_path_write = tmp_path_write, tmp_path_read
    print(f"XY drift shape: {xy_drift.shape}")

    # Step 3: Z correction
    print("\n[3/5] Detecting Z drift...")
    print(f"Data shape before Z drift detection: {data.shape}")
    print(f"Data[ref_channel] shape: {data[ref_channel].shape}")
    z_drift = get_z_drift(tmp_data, ref_channel)
    print(f"Z drift shape: {z_drift.shape}")
    tmp_data = apply_z_drift(tmp_data, z_drift)
    tmp_data = write_tmp_data_to_disk(tmp_path_write, tmp_data, new_shape)
    # Swap paths for next iteration
    tmp_path_read, tmp_path_write = tmp_path_write, tmp_path_read
    print(f"Z drift shape: {z_drift.shape}")

    # Step 4: 3D Rotation correction - Sequential estimation and application
    print("\n[4/5] Detecting and applying 3D rotations sequentially...")

    # Alpha (XY plane) rotation
    print("  - Detecting XY plane rotation (alpha)...")
    alpha_xy = get_rotation_alpha(tmp_data, ref_channel)
    print("  - Applying alpha rotation...")
    tmp_data = apply_alpha_drift(tmp_data, alpha_xy)
    tmp_data = write_tmp_data_to_disk(tmp_path_write, tmp_data, new_shape)
    tmp_path_read, tmp_path_write = tmp_path_write, tmp_path_read

    # Beta (ZX plane) rotation - detected on alpha-corrected data
    print("  - Detecting ZX plane rotation (beta)...")
    beta_zx = get_rotation_beta(tmp_data, ref_channel)
    print("  - Applying beta rotation...")
    tmp_data = apply_beta_drift(tmp_data, beta_zx)
    tmp_data = write_tmp_data_to_disk(tmp_path_write, tmp_data, new_shape)
    tmp_path_read, tmp_path_write = tmp_path_write, tmp_path_read

    # Gamma (ZY plane) rotation - detected on alpha+beta-corrected data
    print("  - Detecting ZY plane rotation (gamma)...")
    gamma_zy = get_rotation_gamma(tmp_data, ref_channel)
    print("  - Applying gamma rotation...")
    tmp_data = apply_gamma_drift(tmp_data, gamma_zy)
    tmp_data = write_tmp_data_to_disk(tmp_path_write, tmp_data, new_shape)
    tmp_path_read, tmp_path_write = tmp_path_write, tmp_path_read

    print("  Rotation angles detected:")
    print(f"    XY (alpha): {alpha_xy[:5]} ...")
    print(f"    ZX (beta):  {beta_zx[:5]} ...")
    print(f"    ZY (gamma): {gamma_zy[:5]} ...")

    # Step 5: Export results
    print("\n[5/5] Exporting registered image...")
    registered_data = tmp_data.compute()
    zarr_path = output_dir / "registered.zarr"
    # Determine optimal chunking for final output
    shape = registered_data.shape
    if len(shape) == 5:  # TCZYX
        chunks = (1, 1, shape[2], shape[3], shape[4])
    else:
        chunks = None
    da.from_array(registered_data, chunks=chunks).to_zarr(str(zarr_path), overwrite=True)
    print(f"Registered image saved to: {output_dir / 'registered.zarr'}")

    # Check output files
    print("\n" + "="*60)
    print("TEST RESULTS:")
    print("="*60)
    output_files = [
        "registered.zarr",
        "XY-Drift.svg",
        "Z-Drift.svg",
        "Rotation-Drift.svg",
    ]

    for fname in output_files:
        fpath = output_dir / fname
        if fpath.exists():
            if fpath.is_dir():
                # Calculate directory size for Zarr stores
                size = sum(f.stat().st_size for f in fpath.rglob('*') if f.is_file()) / 1024  # KB
            else:
                size = fpath.stat().st_size / 1024  # KB
            print(f"✓ {fname:25} ({size:>10.1f} KB)")
        else:
            print(f"✗ {fname:25} (NOT FOUND)")

    print("="*60)
    print("✓ Test run completed successfully!")

if __name__ == "__main__":
    main()
