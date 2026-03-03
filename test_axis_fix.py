#!/usr/bin/env python3
"""Quick test of axis parsing fix for TZYX."""

import numpy as np
import dask.array as da
import sys
sys.path.insert(0, 'src')

from napari_fast4dreg._widget import convert_to_ctzyx, revert_to_original_axis_order

# Test 1: TZYX input
print("Test 1: TZYX input...")
tzyx_data = np.random.rand(10, 50, 512, 512)  # (T, Z, Y, X)
print(f"  Input shape: {tzyx_data.shape}")
print(f"  Input axis order: TZYX")

try:
    ctzyx_data, single_ch, orig_ndim = convert_to_ctzyx(tzyx_data, "TZYX")
    print(f"  ✓ Converted to CTZYX: {ctzyx_data.shape}")
    print(f"    Single channel: {single_ch}, original ndim: {orig_ndim}")
    
    # Test reversion
    reverted = revert_to_original_axis_order(ctzyx_data, "TZYX")
    print(f"  ✓ Reverted to TZYX: {reverted.shape}")
    
    # Verify shape
    assert reverted.shape == tzyx_data.shape, f"Shape mismatch: {reverted.shape} != {tzyx_data.shape}"
    print("  ✓ Shape matches original!")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: ZYX input
print("\nTest 2: ZYX input (3D)...")
zyx_data = np.random.rand(50, 512, 512)  # (Z, Y, X)
print(f"  Input shape: {zyx_data.shape}")
print(f"  Input axis order: ZYX")

try:
    ctzyx_data, single_ch, orig_ndim = convert_to_ctzyx(zyx_data, "ZYX")
    print(f"  ✓ Converted to CTZYX: {ctzyx_data.shape}")
    print(f"    Single channel: {single_ch}, original ndim: {orig_ndim}")
    
    # Test reversion
    reverted = revert_to_original_axis_order(ctzyx_data, "ZYX")
    print(f"  ✓ Reverted to ZYX: {reverted.shape}")
    
    # Verify shape
    assert reverted.shape == zyx_data.shape, f"Shape mismatch: {reverted.shape} != {zyx_data.shape}"
    print("  ✓ Shape matches original!")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: CTZYX input
print("\nTest 3: CTZYX input (5D)...")
ctzyx_data = np.random.rand(2, 10, 50, 512, 512)  # (C, T, Z, Y, X)
print(f"  Input shape: {ctzyx_data.shape}")
print(f"  Input axis order: CTZYX")

try:
    processed, single_ch, orig_ndim = convert_to_ctzyx(ctzyx_data, "CTZYX")
    print(f"  ✓ Converted to CTZYX: {processed.shape}")
    print(f"    Single channel: {single_ch}, original ndim: {orig_ndim}")
    
    # Test reversion
    reverted = revert_to_original_axis_order(processed, "CTZYX")
    print(f"  ✓ Reverted to CTZYX: {reverted.shape}")
    
    # Verify shape
    assert reverted.shape == ctzyx_data.shape, f"Shape mismatch: {reverted.shape} != {ctzyx_data.shape}"
    print("  ✓ Shape matches original!")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: TYX input (no Z, no C)
print("\nTest 4: TYX input (no Z, no C)...")
tyx_data = np.random.rand(10, 512, 512)  # (T, Y, X)
print(f"  Input shape: {tyx_data.shape}")
print(f"  Input axis order: TYX")

try:
    ctzyx_data, single_ch, orig_ndim = convert_to_ctzyx(tyx_data, "TYX")
    print(f"  ✓ Converted to CTZYX: {ctzyx_data.shape}")
    print(f"    Single channel: {single_ch}, original ndim: {orig_ndim}")
    
    # Test reversion
    reverted = revert_to_original_axis_order(ctzyx_data, "TYX")
    print(f"  ✓ Reverted to TYX: {reverted.shape}")
    
    # Verify shape
    assert reverted.shape == tyx_data.shape, f"Shape mismatch: {reverted.shape} != {tyx_data.shape}"
    print("  ✓ Shape matches original!")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("All tests completed!")
