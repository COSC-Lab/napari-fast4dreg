"""
Tests for core registration functions.
"""
import numpy as np
import pytest
import dask.array as da

from napari_fast4dreg._fast4Dreg_functions import (
    apply_projection,
    get_reference_data,
    get_xy_drift,
    apply_xy_drift,
    get_z_drift,
    apply_z_drift,
)


@pytest.fixture
def test_image_4d():
    """Create a small 4D test image (TZYX format)."""
    np.random.seed(42)
    # Create image with slight drift to detect
    base = np.random.randint(50, 200, (3, 8, 32, 32), dtype=np.uint8).astype(np.float32)
    return da.from_array(base, chunks=(1, -1, -1, -1))


@pytest.fixture
def test_image_5d():
    """Create a small 5D test image (CTZYX format)."""
    np.random.seed(42)
    base = np.random.randint(50, 200, (2, 3, 8, 32, 32), dtype=np.uint8).astype(np.float32)
    return da.from_array(base, chunks=(1, 1, -1, -1, -1))


def test_apply_projection_average(test_image_4d):
    """Test average projection."""
    result = apply_projection(test_image_4d, axis=1, projection_type='average')
    
    # Should reduce Z dimension
    assert result.shape == (3, 32, 32)
    
    # Check it's actually an average
    expected = test_image_4d.mean(axis=1)
    np.testing.assert_array_almost_equal(result.compute(), expected.compute())


def test_apply_projection_max(test_image_4d):
    """Test max projection."""
    result = apply_projection(test_image_4d, axis=1, projection_type='max')
    
    assert result.shape == (3, 32, 32)
    expected = test_image_4d.max(axis=1)
    np.testing.assert_array_equal(result.compute(), expected.compute())


def test_apply_projection_median(test_image_4d):
    """Test median projection."""
    result = apply_projection(test_image_4d, axis=1, projection_type='median')
    
    assert result.shape == (3, 32, 32)


def test_apply_projection_min(test_image_4d):
    """Test min projection."""
    result = apply_projection(test_image_4d, axis=1, projection_type='min')
    
    assert result.shape == (3, 32, 32)
    expected = test_image_4d.min(axis=1)
    np.testing.assert_array_equal(result.compute(), expected.compute())


def test_apply_projection_invalid_type(test_image_4d):
    """Test that invalid projection type raises error."""
    with pytest.raises(ValueError, match="Unknown projection type"):
        apply_projection(test_image_4d, axis=1, projection_type='invalid')


def test_get_reference_data_single_channel(test_image_5d):
    """Test getting single reference channel."""
    ref_data = get_reference_data(test_image_5d, ref_channel=0)
    
    # Should return first channel
    assert ref_data.shape == test_image_5d[0].shape
    np.testing.assert_array_equal(ref_data.compute(), test_image_5d[0].compute())


def test_get_reference_data_multi_channel(test_image_5d):
    """Test getting multiple reference channels."""
    ref_data = get_reference_data(test_image_5d, ref_channel="0,1", normalize_channels=False)
    
    # Should return sum of both channels
    assert ref_data.shape == test_image_5d[0].shape
    expected = test_image_5d[0] + test_image_5d[1]
    np.testing.assert_array_equal(ref_data.compute(), expected.compute())


def test_get_reference_data_multi_channel_normalized(test_image_5d):
    """Test getting multiple reference channels with normalization."""
    ref_data = get_reference_data(test_image_5d, ref_channel="0,1", normalize_channels=True)
    
    # Should return normalized sum
    assert ref_data.shape == test_image_5d[0].shape
    
    # Result should be normalized (not just raw sum)
    # The values should be different from simple sum
    simple_sum = test_image_5d[0] + test_image_5d[1]
    assert not np.array_equal(ref_data.compute(), simple_sum.compute())


def test_get_reference_data_space_separated(test_image_5d):
    """Test getting multiple reference channels with space-separated format."""
    ref_data = get_reference_data(test_image_5d, ref_channel="0 1", normalize_channels=False)
    
    # Should return sum of both channels (same as comma-separated)
    assert ref_data.shape == test_image_5d[0].shape
    expected = test_image_5d[0] + test_image_5d[1]
    np.testing.assert_array_equal(ref_data.compute(), expected.compute())
    
    # Test that comma and space formats give identical results
    ref_data_comma = get_reference_data(test_image_5d, ref_channel="0,1", normalize_channels=False)
    np.testing.assert_array_equal(ref_data.compute(), ref_data_comma.compute())


def test_get_xy_drift_shape(test_image_5d):
    """Test that XY drift detection returns correct shape."""
    xy_drift = get_xy_drift(
        test_image_5d,
        ref_channel=0,
        projection_type='average',
        reference_mode='relative'
    )
    
    # Should return (T, 2) array for T timepoints
    num_timepoints = test_image_5d.shape[1]
    assert xy_drift.shape == (num_timepoints, 2)


def test_get_xy_drift_relative_mode(test_image_5d):
    """Test XY drift in relative mode."""
    xy_drift = get_xy_drift(
        test_image_5d,
        ref_channel=0,
        projection_type='average',
        reference_mode='relative'
    )
    
    # First frame should have zero drift in relative mode
    assert xy_drift[0, 0] == 0.0
    assert xy_drift[0, 1] == 0.0


def test_get_xy_drift_first_frame_mode(test_image_5d):
    """Test XY drift in first_frame mode."""
    xy_drift = get_xy_drift(
        test_image_5d,
        ref_channel=0,
        projection_type='average',
        reference_mode='first_frame'
    )
    
    # First frame should have zero drift
    assert xy_drift[0, 0] == 0.0
    assert xy_drift[0, 1] == 0.0


def test_apply_xy_drift_shape(test_image_5d):
    """Test that applying XY drift preserves shape."""
    xy_drift = np.array([[0, 0], [1, 0], [0, 1]])
    
    result = apply_xy_drift(test_image_5d, xy_drift)
    
    assert result.shape == test_image_5d.shape


def test_get_z_drift_shape(test_image_5d):
    """Test that Z drift detection returns correct shape."""
    z_drift = get_z_drift(
        test_image_5d,
        ref_channel=0,
        projection_type='average',
        reference_mode='relative'
    )
    
    # Z drift returns (T, 2) array where second column is always 0
    num_timepoints = test_image_5d.shape[1]
    assert z_drift.shape == (num_timepoints, 2)
    # Second column should be all zeros (only Z drift, no Y)
    assert np.all(z_drift[:, 1] == 0)


def test_get_z_drift_relative_mode(test_image_5d):
    """Test Z drift in relative mode."""
    z_drift = get_z_drift(
        test_image_5d,
        ref_channel=0,
        projection_type='average',
        reference_mode='relative'
    )
    
    # First frame should have zero drift
    assert z_drift[0, 0] == 0.0
    assert z_drift[0, 1] == 0.0


def test_apply_z_drift_shape(test_image_5d):
    """Test that applying Z drift preserves shape."""
    # Z drift is (T, 2) format where only first column matters
    z_drift = np.array([[0, 0], [1, 0], [-1, 0]])
    
    result = apply_z_drift(test_image_5d, z_drift)
    
    # Shape might change slightly due to interpolation, but should be close
    assert result.shape[0] == test_image_5d.shape[0]  # Channels unchanged
    assert result.shape[1] == test_image_5d.shape[1]  # Time unchanged


def test_projection_types_consistency():
    """Test that all projection types work consistently."""
    data = da.from_array(np.random.rand(5, 10, 64, 64).astype(np.float32), chunks=(1, -1, -1, -1))
    
    for proj_type in ['average', 'max', 'median', 'min']:
        result = apply_projection(data, axis=1, projection_type=proj_type)
        assert result.shape == (5, 64, 64)
        assert not np.any(np.isnan(result.compute()))


def test_reference_modes_consistency(test_image_5d):
    """Test that both reference modes produce valid results."""
    for ref_mode in ['relative', 'first_frame']:
        xy_drift = get_xy_drift(
            test_image_5d,
            ref_channel=0,
            projection_type='average',
            reference_mode=ref_mode
        )
        
        assert xy_drift.shape == (3, 2)
        assert xy_drift[0, 0] == 0.0  # First frame always has zero drift
        assert xy_drift[0, 1] == 0.0


def test_multi_channel_reference_consistency(test_image_5d):
    """Test that multi-channel reference works with all modes."""
    for normalize in [False, True]:
        ref_data = get_reference_data(
            test_image_5d,
            ref_channel="0,1",
            normalize_channels=normalize
        )
        
        assert ref_data.shape == test_image_5d[0].shape
        assert not np.any(np.isnan(ref_data.compute()))
