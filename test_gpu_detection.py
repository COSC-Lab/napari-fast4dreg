#!/usr/bin/env python
"""
Test script for GPU detection and device selection.
Verifies that napari-fast4dreg correctly detects and selects GPUs.
"""

import sys
from pathlib import Path

# Add src to path for development testing
sys.path.insert(0, str(Path(__file__).parent / "src" / "napari_fast4dreg"))

print("="*70)
print("GPU Detection Test for napari-fast4dreg")
print("="*70)

# Import the module - this triggers automatic GPU detection
print("\nImporting napari_fast4dreg._fast4Dreg_functions...")
print("-" * 70)

import _fast4Dreg_functions as f4d

print("-" * 70)
print("\n✓ Import complete")

# Check the results
print("\nDetection Results:")
print("="*70)
print(f"  pyclesperanto available: {f4d.PYCLESPERANTO_AVAILABLE}")
print(f"  GPU acceleration enabled: {f4d.USE_GPU_ACCELERATION}")
print(f"  Backend: {f4d.get_gpu_info()}")
print("="*70)

if f4d.PYCLESPERANTO_AVAILABLE:
    print("\n✓ pyclesperanto is installed")

    # Try to get more GPU details
    try:
        import pyclesperanto_prototype as cle
        print("\nAvailable OpenCL devices:")
        print("-" * 70)
        devices = cle.available_device_names()
        for i, device in enumerate(devices, 1):
            marker = "  [SELECTED] " if f4d.USE_GPU_ACCELERATION and device in f4d.GPU_INFO else "           "
            print(f"{marker}{i}. {device}")

        if f4d.USE_GPU_ACCELERATION:
            print("\n✓ GPU acceleration ENABLED automatically")
            print(f"  Selected device: {f4d.GPU_INFO}")
        else:
            print("\n✗ GPU acceleration NOT enabled (check logs above for reason)")

    except Exception as e:
        print(f"\n⚠ Error getting GPU details: {e}")
else:
    print("\n✗ pyclesperanto NOT installed")
    print("\nTo enable GPU acceleration:")
    print("  pip install pyclesperanto-prototype")

print("\n" + "="*70)
print("Test Functions")
print("="*70)

# Test manual control
print("\n1. Testing manual GPU disable...")
result = f4d.set_gpu_acceleration(False)
print(f"   Result: {result} (should be False)")
print(f"   Backend: {f4d.get_gpu_info()}")

if f4d.PYCLESPERANTO_AVAILABLE:
    print("\n2. Testing manual GPU enable...")
    result = f4d.set_gpu_acceleration(True)
    print(f"   Result: {result}")
    print(f"   Backend: {f4d.get_gpu_info()}")

print("\n" + "="*70)
print("Test Complete")
print("="*70)
