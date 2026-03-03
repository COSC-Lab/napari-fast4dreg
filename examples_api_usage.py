#!/usr/bin/env python
"""
Example: Using napari-fast4dreg programmatically via API.

This script demonstrates how to integrate Fast4DReg into your
workflows without the napari GUI.

Key Features:
- XY, Z, and 3D rotation drift correction
- Sequential rotation correction (XY → ZX → ZY) for improved accuracy
- Flexible reference channel selection (single or multiple channels)
- Progress callbacks for integration
- Efficient out-of-memory processing with dask/zarr
"""

from pathlib import Path

import numpy as np

# Import the API
from napari_fast4dreg import register_image_from_file


def example_1_basic():
    """Example 1: Basic registration with numpy array."""
    print("="*60)
    print("Example 1: Basic registration")
    print("="*60)

    # Create or load your image data (CTZYX format)
    # For this example, we'll load from file
    example_dir = Path(__file__).parent / "example_files"

    # Option A: Load and register in one step
    result = register_image_from_file(
        example_dir / "xtitched_organoid_timelapse_ch0_cytosol_ch1_nuclei.tif",
        axis_order="TZCYX",  # ImageJ format
        ref_channel=1,  # Use channel 1 (nuclei)
        output_dir=example_dir / "api_output_example1",
        correct_xy=True,
        correct_z=True,
        correct_rotation=True,
    )

    # Access results
    registered = result['registered_image']
    print(f"Registered image shape: {registered.shape}")
    print(f"Output saved to: {result['output_path']}")
    print(f"XY drift shape: {result['xy_drift'].shape}")
    print(f"Z drift shape: {result['z_drift'].shape}")
    print()


def example_2_with_progress():
    """Example 2: Registration with progress tracking."""
    print("="*60)
    print("Example 2: Registration with progress tracking")
    print("="*60)

    # Define progress callback
    def show_progress(message):
        print(f"  [PROGRESS] {message}")

    example_dir = Path(__file__).parent / "example_files"

    result = register_image_from_file(
        example_dir / "xtitched_organoid_timelapse_ch0_cytosol_ch1_nuclei.tif",
        axis_order="TZCYX",
        ref_channel=1,
        output_dir=example_dir / "api_output_example2",
        correct_xy=True,
        correct_z=True,
        correct_rotation=False,  # Skip rotation for speed
        progress_callback=show_progress,
    )

    print(f"\nCompleted! Registered image saved to: {result['output_path']}")
    print()


def example_3_selective_corrections():
    """Example 3: Selective corrections only."""
    print("="*60)
    print("Example 3: Only XY drift correction")
    print("="*60)

    example_dir = Path(__file__).parent / "example_files"

    # Only correct XY drift, skip Z and rotation
    result = register_image_from_file(
        example_dir / "xtitched_organoid_timelapse_ch0_cytosol_ch1_nuclei.tif",
        axis_order="TZCYX",
        ref_channel=1,
        output_dir=example_dir / "api_output_example3",
        correct_xy=True,
        correct_z=False,  # Skip Z
        correct_rotation=False,  # Skip rotation
        progress_callback=lambda msg: print(f"  {msg}"),
    )

    print(f"\nXY drift detected: {result['xy_drift'][:5]}")
    print(f"Output: {result['output_path']}")
    print()


def example_4_multi_channel_reference():
    """Example 4: Multi-channel reference with normalization."""
    print("="*60)
    print("Example 4: Multi-channel reference")
    print("="*60)

    example_dir = Path(__file__).parent / "example_files"

    # Use multiple channels as reference
    result = register_image_from_file(
        example_dir / "xtitched_organoid_timelapse_ch0_cytosol_ch1_nuclei.tif",
        axis_order="TZCYX",
        ref_channel="0,1",  # Use both channels
        normalize_channels=True,  # Normalize before summing
        projection_type='max',  # Use max projection
        reference_mode='first_frame',  # Compare to first frame
        output_dir=example_dir / "api_output_example4",
        progress_callback=lambda msg: print(f"  {msg}"),
    )

    print("\nRegistered with multi-channel reference")
    print(f"Output: {result['output_path']}")
    print()


def example_5_rotation_correction():
    """Example 5: Understanding sequential rotation correction."""
    print("="*60)
    print("Example 5: Sequential Rotation Correction")
    print("="*60)

    example_dir = Path(__file__).parent / "example_files"

    print("\nFast4DReg applies 3D rotation correction sequentially:")
    print("  1. Alpha (XY plane) - detected and applied first")
    print("  2. Beta (ZX plane) - detected on alpha-corrected data")
    print("  3. Gamma (ZY plane) - detected on alpha+beta-corrected data")
    print("\nThis sequential approach improves accuracy when multiple")
    print("rotation components are present.")
    print()

    # Run with rotation correction enabled
    result = register_image_from_file(
        example_dir / "xtitched_organoid_timelapse_ch0_cytosol_ch1_nuclei.tif",
        axis_order="TZCYX",
        ref_channel=1,
        output_dir=example_dir / "api_output_example5",
        correct_xy=True,
        correct_z=True,
        correct_rotation=True,  # Sequential XY→ZX→ZY rotation
        progress_callback=lambda msg: print(f"  {msg}"),
    )

    # Analyze rotation components
    print("\n  Rotation angles detected:")
    print(f"    XY (alpha): mean={result['rotation_xy'].mean():.3f}°, max={result['rotation_xy'].max():.3f}°")
    print(f"    ZX (beta):  mean={result['rotation_zx'].mean():.3f}°, max={result['rotation_zx'].max():.3f}°")
    print(f"    ZY (gamma): mean={result['rotation_zy'].mean():.3f}°, max={result['rotation_zy'].max():.3f}°")
    print(f"\n  Output: {result['output_path']}")
    print()


def example_6_integration_workflow():
    """Example 6: Integration into a larger workflow."""
    print("="*60)
    print("Example 6: Integration into workflow")
    print("="*60)

    example_dir = Path(__file__).parent / "example_files"

    # Step 1: Your preprocessing
    print("Step 1: Preprocessing...")
    # (Your preprocessing code here)

    # Step 2: Registration
    print("Step 2: Running Fast4DReg registration...")
    result = register_image_from_file(
        example_dir / "xtitched_organoid_timelapse_ch0_cytosol_ch1_nuclei.tif",
        axis_order="TZCYX",
        ref_channel=1,
        output_dir=example_dir / "api_output_example6",
        correct_xy=True,
        correct_z=True,
        correct_rotation=True,
        keep_temp_files=False,  # Clean up temp files
        return_drifts=True,  # Get drift data
    )

    registered_image = result['registered_image']

    # Step 3: Your postprocessing
    print("Step 3: Postprocessing...")
    # (Your postprocessing code here)

    # Step 4: Analysis using drift information
    print("Step 4: Analyzing drift patterns...")
    xy_drift = result['xy_drift']
    total_drift = np.sqrt(xy_drift[:, 0]**2 + xy_drift[:, 1]**2)
    print(f"  Max XY drift: {total_drift.max():.2f} pixels")
    print(f"  Mean XY drift: {total_drift.mean():.2f} pixels")

    if 'z_drift' in result:
        z_drift = result['z_drift']
        print(f"  Max Z drift: {np.abs(z_drift).max():.2f} pixels")
        print(f"  Mean Z drift: {np.abs(z_drift).mean():.2f} pixels")

    print()


def example_7_gpu_acceleration():
    """Example 7: GPU acceleration - automatic detection and manual control."""
    print("="*60)
    print("Example 7: GPU Acceleration (Automatic Detection)")
    print("="*60)

    from napari_fast4dreg import get_gpu_info

    example_dir = Path(__file__).parent / "example_files"

    print("\nGPU acceleration is AUTOMATICALLY DETECTED and enabled on import.")
    print("NVIDIA GPUs are preferred over Intel GPUs.")
    print("Requires: pip install pyclesperanto-prototype")
    print()

    # Check current GPU status
    print(f"Current backend: {get_gpu_info()}")

    # Check if GPU is already enabled from automatic detection
    from napari_fast4dreg._fast4Dreg_functions import USE_GPU_ACCELERATION

    if USE_GPU_ACCELERATION:
        print("✓ GPU acceleration was automatically enabled!")
        print("\nRunning registration with GPU acceleration...")

        import time
        start_time = time.time()

        result = register_image_from_file(
            example_dir / "xtitched_organoid_timelapse_ch0_cytosol_ch1_nuclei.tif",
            axis_order="TZCYX",
            ref_channel=1,
            output_dir=example_dir / "api_output_example7_gpu",
            correct_xy=True,
            correct_z=True,
            correct_rotation=True,
            progress_callback=lambda msg: print(f"  {msg}"),
        )

        gpu_time = time.time() - start_time
        print(f"\n  GPU processing time: {gpu_time:.1f}s")
        print(f"  GPU device: {get_gpu_info()}")
        print(f"  Output: {result['output_path']}")

        # Show how to manually disable GPU
        print("\n  You can manually disable GPU if needed:")
        print("  set_gpu_acceleration(False)")
        print("  Typical GPU speedup: 5-10x for transformation operations")
    else:
        print("✗ GPU acceleration not available (CPU mode)")
        print(f"  Backend: {get_gpu_info()}")
        print("\n  To enable GPU acceleration:")
        print("  1. Install: pip install pyclesperanto-prototype")
        print("  2. Ensure OpenCL-compatible GPU is available")
        print("  3. Restart Python - GPU will be auto-detected")
        print("\n  Or manually enable: set_gpu_acceleration(True)")
        print("  Requires OpenCL-compatible GPU (NVIDIA, AMD, or Intel)")

    print()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("napari-fast4dreg API Examples")
    print("="*60 + "\n")

    # Run examples
    try:
        example_1_basic()
    except Exception as e:
        print(f"Example 1 failed: {e}\n")

    try:
        example_2_with_progress()
    except Exception as e:
        print(f"Example 2 failed: {e}\n")

    try:
        example_3_selective_corrections()
    except Exception as e:
        print(f"Example 3 failed: {e}\n")

    try:
        example_4_multi_channel_reference()
    except Exception as e:
        print(f"Example 4 failed: {e}\n")

    try:
        example_5_rotation_correction()
    except Exception as e:
        print(f"Example 5 failed: {e}\n")

    try:
        example_6_integration_workflow()
    except Exception as e:
        print(f"Example 6 failed: {e}\n")

    try:
        example_7_gpu_acceleration()
    except Exception as e:
        print(f"Example 7 failed: {e}\n")

    print("="*60)
    print("All examples completed!")
    print("="*60)
