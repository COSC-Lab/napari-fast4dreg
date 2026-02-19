#!/usr/bin/env python
"""
Example: Using napari-fast4dreg programmatically via API.

This script demonstrates how to integrate Fast4DReg into your
workflows without the napari GUI.
"""

import numpy as np
from pathlib import Path

# Import the API
from napari_fast4dreg import register_image, register_image_from_file


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
    
    print(f"\nRegistered with multi-channel reference")
    print(f"Output: {result['output_path']}")
    print()


def example_5_integration_workflow():
    """Example 5: Integration into a larger workflow."""
    print("="*60)
    print("Example 5: Integration into workflow")
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
        output_dir=example_dir / "api_output_example5",
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
        example_5_integration_workflow()
    except Exception as e:
        print(f"Example 5 failed: {e}\n")
    
    print("="*60)
    print("All examples completed!")
    print("="*60)
