"""
Tests for the programmatic API (register_image, register_image_from_file).

Test categories:
1. Synthetic data tests - Fast unit tests with small generated images
2. Real data tests - Integration tests using actual example files
   - Single-channel file: TZYX format (21 timepoints, 64 z-slices, 128x128)
   - Multi-channel file: TZCYX format (21 timepoints, 64 z-slices, 2 channels, 128x128)
   
Real data tests are marked with @pytest.mark.slow and can be run with:
    pytest -m slow
Or skipped with:
    pytest -m "not slow"
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from napari_fast4dreg.api import register_image, register_image_from_file


@pytest.fixture
def test_image_5d():
    """Create a small 5D test image (CTZYX format)."""
    # Small test image: 2 channels, 3 timepoints, 4 Z-slices, 32x32
    np.random.seed(42)
    image = np.random.randint(0, 255, (2, 3, 4, 32, 32), dtype=np.uint8)
    return image


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix="test_fast4dreg_")
    yield Path(temp_dir)
    # Cleanup after test
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)


def test_register_image_basic(test_image_5d, temp_output_dir):
    """Test basic registration with minimal corrections."""
    result = register_image(
        test_image_5d,
        axis_order="CTZYX",
        ref_channel=0,
        output_dir=temp_output_dir,
        correct_xy=True,
        correct_z=False,
        correct_rotation=False,
        crop_output=False,
        return_drifts=True,
        keep_temp_files=False
    )

    # Check result structure
    assert 'registered_image' in result
    assert 'xy_drift' in result
    assert 'output_path' in result

    # Check registered image shape matches input
    assert result['registered_image'].shape == test_image_5d.shape

    # Check XY drift shape (should be Tx2)
    assert result['xy_drift'].shape == (3, 2)

    # Check output file exists
    assert result['output_path'].exists()

    # Check temp files were cleaned up
    assert not (temp_output_dir / "tmp_data_1.zarr").exists()
    assert not (temp_output_dir / "tmp_data_2.zarr").exists()


def test_register_image_all_corrections(test_image_5d, temp_output_dir):
    """Test with all corrections enabled."""
    result = register_image(
        test_image_5d,
        ref_channel=0,
        output_dir=temp_output_dir,
        correct_xy=True,
        correct_z=True,
        correct_rotation=True,
        crop_output=False,
        return_drifts=True
    )

    # Check all drift data is present
    assert 'xy_drift' in result
    assert 'z_drift' in result
    assert 'rotation_xy' in result
    assert 'rotation_zx' in result
    assert 'rotation_zy' in result

    # Check drift shapes
    assert result['xy_drift'].shape == (3, 2)
    assert result['z_drift'].shape == (3, 2)  # Z drift returns (T, 2) where second column is 0
    assert result['rotation_xy'].shape == (3,)
    assert result['rotation_zx'].shape == (3,)
    assert result['rotation_zy'].shape == (3,)


def test_register_image_progress_callback(test_image_5d, temp_output_dir):
    """Test that progress callback is called."""
    progress_messages = []

    def collect_progress(message):
        progress_messages.append(message)

    result = register_image(
        test_image_5d,
        ref_channel=0,
        output_dir=temp_output_dir,
        correct_xy=True,
        correct_z=False,
        correct_rotation=False,
        progress_callback=collect_progress
    )

    # Check that progress was reported
    assert len(progress_messages) > 0
    assert any("Converting to dask array" in msg or "XY" in msg for msg in progress_messages)


def test_register_image_keep_temp_files(test_image_5d, temp_output_dir):
    """Test that temporary files are kept when requested."""
    result = register_image(
        test_image_5d,
        ref_channel=0,
        output_dir=temp_output_dir,
        correct_xy=True,
        correct_z=False,
        correct_rotation=False,
        keep_temp_files=True
    )

    # Check temp files exist
    assert (temp_output_dir / "tmp_data_1.zarr").exists() or \
           (temp_output_dir / "tmp_data_2.zarr").exists()


def test_register_image_no_drifts(test_image_5d, temp_output_dir):
    """Test with return_drifts=False."""
    result = register_image(
        test_image_5d,
        ref_channel=0,
        output_dir=temp_output_dir,
        correct_xy=True,
        correct_z=False,
        correct_rotation=False,
        return_drifts=False
    )

    # Check that registered image exists but drift data doesn't
    assert 'registered_image' in result
    assert 'output_path' in result
    assert 'xy_drift' not in result


def test_register_image_multi_channel_reference(test_image_5d, temp_output_dir):
    """Test with multiple reference channels."""
    result = register_image(
        test_image_5d,
        ref_channel="0,1",  # Use both channels
        normalize_channels=True,
        output_dir=temp_output_dir,
        correct_xy=True,
        correct_z=False,
        correct_rotation=False,
        return_drifts=True
    )

    assert 'registered_image' in result
    assert result['registered_image'].shape == test_image_5d.shape


def test_register_image_projection_types(test_image_5d, temp_output_dir):
    """Test different projection types."""
    for proj_type in ['average', 'max', 'median', 'min']:
        output_subdir = temp_output_dir / proj_type
        output_subdir.mkdir(exist_ok=True)

        result = register_image(
            test_image_5d,
            ref_channel=0,
            projection_type=proj_type,
            output_dir=output_subdir,
            correct_xy=True,
            correct_z=False,
            correct_rotation=False,
        )

        assert 'registered_image' in result


def test_register_image_reference_modes(test_image_5d, temp_output_dir):
    """Test different reference modes."""
    for ref_mode in ['relative', 'first_frame']:
        output_subdir = temp_output_dir / ref_mode
        output_subdir.mkdir(exist_ok=True)

        result = register_image(
            test_image_5d,
            ref_channel=0,
            reference_mode=ref_mode,
            output_dir=output_subdir,
            correct_xy=True,
            correct_z=False,
            correct_rotation=False,
        )

        assert 'registered_image' in result


def test_register_image_invalid_shape():
    """Test that invalid axis order raises error."""
    # Valid 4D image but with invalid axis order
    invalid_image = np.random.randint(0, 255, (3, 4, 32, 32), dtype=np.uint8)

    # Should raise error for invalid axis order (missing Y or X)
    with pytest.raises(ValueError, match="Invalid axis order"):
        register_image(
            invalid_image,
            axis_order="TZDC",  # Invalid: no Y or X
            ref_channel=0,
            output_dir="./test_output"
        )


def test_register_image_tzyx_format(temp_output_dir):
    """Test registration with TZYX format (4D, single channel)."""
    # TZYX format: 3 timepoints, 4 Z-slices, 32x32
    image_tzyx = np.random.randint(0, 255, (3, 4, 32, 32), dtype=np.uint8)
    
    result = register_image(
        image_tzyx,
        axis_order="TZYX",
        ref_channel=0,
        output_dir=temp_output_dir,
        correct_xy=True,
        correct_z=False,
        correct_rotation=False,
        return_drifts=True,
    )
    
    # Output shape should match input
    assert result['registered_image'].shape == image_tzyx.shape
    assert "xy_drift" in result


def test_register_image_zyx_format(temp_output_dir):
    """Test registration with ZYX format (3D, single timepoint + channel)."""
    # ZYX format: 4 Z-slices, 32x32
    image_zyx = np.random.randint(0, 255, (4, 32, 32), dtype=np.uint8)
    
    result = register_image(
        image_zyx,
        axis_order="ZYX",
        ref_channel=0,
        output_dir=temp_output_dir,
        correct_xy=True,
        correct_z=False,
        correct_rotation=False,
        return_drifts=True,
    )
    
    # Output shape should match input
    assert result['registered_image'].shape == image_zyx.shape


def test_register_image_zcyx_format(temp_output_dir):
    """Test registration with ZCYX format (4D, single timepoint)."""
    # ZCYX format: 2 Z-slices, 2 channels, 32x32
    image_zcyx = np.random.randint(0, 255, (2, 2, 32, 32), dtype=np.uint8)
    
    result = register_image(
        image_zcyx,
        axis_order="ZCYX",
        ref_channel=0,
        output_dir=temp_output_dir,
        correct_xy=True,
        correct_z=False,
        correct_rotation=False,
        return_drifts=True,
    )
    
    # Output shape should match input
    assert result['registered_image'].shape == image_zcyx.shape


def test_register_image_from_file_tiff(temp_output_dir):
    """Test loading and registering from TIFF file."""
    # Create a test TIFF file
    test_image = np.random.randint(0, 255, (3, 4, 2, 32, 32), dtype=np.uint8)
    tiff_path = temp_output_dir / "test_image.tif"

    import tifffile
    tifffile.imwrite(tiff_path, test_image)

    # Register from file
    output_subdir = temp_output_dir / "output"
    result = register_image_from_file(
        tiff_path,
        axis_order="TZCYX",
        ref_channel=0,
        output_dir=output_subdir,
        correct_xy=True,
        correct_z=False,
        correct_rotation=False,
    )

    assert 'registered_image' in result
    assert result['registered_image'].shape[0] == 2  # Channels after reordering


def test_register_image_from_file_npy(temp_output_dir):
    """Test loading and registering from NPY file."""
    # Create a test NPY file
    test_image = np.random.randint(0, 255, (2, 3, 4, 32, 32), dtype=np.uint8)
    npy_path = temp_output_dir / "test_image.npy"
    np.save(npy_path, test_image)

    # Register from file
    output_subdir = temp_output_dir / "output"
    result = register_image_from_file(
        npy_path,
        axis_order="CTZYX",
        ref_channel=0,
        output_dir=output_subdir,
        correct_xy=True,
        correct_z=False,
        correct_rotation=False,
    )

    assert 'registered_image' in result
    assert result['registered_image'].shape == test_image.shape


def test_api_aliases():
    """Test that API aliases exist and point to correct functions."""
    from napari_fast4dreg import fast4dreg, register

    # Check aliases exist
    assert fast4dreg is register_image
    assert register is register_image


# ============================================================================
# REAL DATA TESTS - Using actual example files
# ============================================================================

@pytest.fixture
def example_files_dir():
    """Get path to example_files directory."""
    # Assuming tests are run from package root or within src structure
    test_dir = Path(__file__).parent
    example_dir = test_dir.parent.parent.parent.parent / "example_files"

    # Alternative: try to find it relative to common locations
    if not example_dir.exists():
        example_dir = Path.cwd() / "example_files"

    if not example_dir.exists():
        pytest.skip("example_files directory not found")

    return example_dir


@pytest.fixture
def single_channel_file(example_files_dir):
    """Get single-channel example file (TZYX format)."""
    file_path = example_files_dir / "xtitched_organoid_timelapse_ch0_cytosol_ch1_nuclei-1.tif"
    if not file_path.exists():
        pytest.skip(f"Single-channel example file not found: {file_path}")
    return file_path


@pytest.fixture
def multi_channel_file(example_files_dir):
    """Get multi-channel example file (TZCYX format)."""
    file_path = example_files_dir / "xtitched_organoid_timelapse_ch0_cytosol_ch1_nuclei.tif"
    if not file_path.exists():
        pytest.skip(f"Multi-channel example file not found: {file_path}")
    return file_path


@pytest.mark.slow
def test_real_data_single_channel_basic(single_channel_file, temp_output_dir):
    """Test registration with real single-channel data - basic XY correction only."""
    result = register_image_from_file(
        single_channel_file,
        axis_order="TZYX",  # Single channel: T, Z, Y, X
        ref_channel=0,
        output_dir=temp_output_dir / "single_channel_basic",
        correct_xy=True,
        correct_z=False,
        correct_rotation=False,
        crop_output=False,
        return_drifts=True,
    )

    # Check result structure
    assert 'registered_image' in result
    assert 'xy_drift' in result
    assert 'output_path' in result

    # Check output shape - should be CTZYX (1, 21, 64, 128, 128)
    registered = result['registered_image']
    assert registered.ndim == 5
    assert registered.shape[0] == 1  # Single channel
    assert registered.shape[1] == 21  # 21 timepoints
    assert registered.shape[2] == 64  # 64 z-slices

    # Check XY drift shape
    assert result['xy_drift'].shape == (21, 2)

    # Verify drift was detected (should not all be zero)
    assert not np.allclose(result['xy_drift'], 0)

    # Check output file exists
    assert result['output_path'].exists()


@pytest.mark.slow
def test_real_data_single_channel_full(single_channel_file, temp_output_dir):
    """Test registration with real single-channel data - all corrections."""
    result = register_image_from_file(
        single_channel_file,
        axis_order="TZYX",
        ref_channel=0,
        output_dir=temp_output_dir / "single_channel_full",
        correct_xy=True,
        correct_z=True,
        correct_rotation=True,  # Sequential XY→ZX→ZY
        crop_output=False,
        projection_type='average',
        reference_mode='relative',
        return_drifts=True,
    )

    # Check all drift data is present
    assert 'xy_drift' in result
    assert 'z_drift' in result
    assert 'rotation_xy' in result
    assert 'rotation_zx' in result
    assert 'rotation_zy' in result

    # Check drift shapes
    assert result['xy_drift'].shape == (21, 2)
    assert result['z_drift'].shape == (21, 2)
    assert result['rotation_xy'].shape == (21,)
    assert result['rotation_zx'].shape == (21,)
    assert result['rotation_zy'].shape == (21,)

    # Check output shape maintained
    registered = result['registered_image']
    assert registered.shape[0] == 1  # Single channel
    assert registered.shape[1] == 21  # 21 timepoints


@pytest.mark.slow
def test_real_data_multi_channel_ch0(multi_channel_file, temp_output_dir):
    """Test registration with real multi-channel data using channel 0."""
    result = register_image_from_file(
        multi_channel_file,
        axis_order="TZCYX",  # ImageJ format: T, Z, Channels, Y, X
        ref_channel=0,  # Use channel 0 (cytosol)
        output_dir=temp_output_dir / "multi_channel_ch0",
        correct_xy=True,
        correct_z=True,
        correct_rotation=False,  # Skip rotation for speed
        crop_output=False,
        return_drifts=True,
    )

    # Check result structure
    assert 'registered_image' in result
    assert 'xy_drift' in result
    assert 'z_drift' in result

    # Check output shape - should be CTZYX (2, 21, 64, 128, 128)
    registered = result['registered_image']
    assert registered.ndim == 5
    assert registered.shape[0] == 2  # 2 channels
    assert registered.shape[1] == 21  # 21 timepoints
    assert registered.shape[2] == 64  # 64 z-slices

    # Check drift shapes
    assert result['xy_drift'].shape == (21, 2)
    assert result['z_drift'].shape == (21, 2)


@pytest.mark.slow
def test_real_data_multi_channel_ch1(multi_channel_file, temp_output_dir):
    """Test registration with real multi-channel data using channel 1 (nuclei)."""
    result = register_image_from_file(
        multi_channel_file,
        axis_order="TZCYX",
        ref_channel=1,  # Use channel 1 (nuclei)
        output_dir=temp_output_dir / "multi_channel_ch1",
        correct_xy=True,
        correct_z=True,
        correct_rotation=False,
        projection_type='max',  # Max projection often good for nuclei
        return_drifts=True,
    )

    # Check results
    assert 'registered_image' in result
    assert result['registered_image'].shape[0] == 2  # 2 channels preserved
    assert result['xy_drift'].shape == (21, 2)

    # Verify drift was detected
    assert not np.allclose(result['xy_drift'], 0)


@pytest.mark.slow
def test_real_data_multi_channel_both_channels(multi_channel_file, temp_output_dir):
    """Test registration using both channels as reference with normalization."""
    result = register_image_from_file(
        multi_channel_file,
        axis_order="TZCYX",
        ref_channel="0,1",  # Use both channels
        normalize_channels=True,  # Normalize before combining
        output_dir=temp_output_dir / "multi_channel_both",
        correct_xy=True,
        correct_z=False,
        correct_rotation=False,
        return_drifts=True,
    )

    # Check results
    assert 'registered_image' in result
    assert result['registered_image'].shape[0] == 2  # Channels preserved
    assert result['xy_drift'].shape == (21, 2)


@pytest.mark.slow
def test_real_data_sequential_rotation_correction(single_channel_file, temp_output_dir):
    """Test that sequential rotation correction works with real data."""
    # Run with progress callback to verify sequential steps
    progress_messages = []

    def track_progress(msg):
        progress_messages.append(msg)

    result = register_image_from_file(
        single_channel_file,
        axis_order="TZYX",
        ref_channel=0,
        output_dir=temp_output_dir / "sequential_rotation",
        correct_xy=True,
        correct_z=True,
        correct_rotation=True,
        progress_callback=track_progress,
        return_drifts=True,
    )

    # Check that rotation steps were executed in sequence
    rotation_steps = [msg for msg in progress_messages if 'rotation' in msg.lower()]

    # Should see alpha, beta, gamma in sequence
    assert any('alpha' in msg.lower() or 'xy plane' in msg.lower() for msg in rotation_steps)
    assert any('beta' in msg.lower() or 'zx plane' in msg.lower() for msg in rotation_steps)
    assert any('gamma' in msg.lower() or 'zy plane' in msg.lower() for msg in rotation_steps)

    # Check all rotation data is present
    assert 'rotation_xy' in result
    assert 'rotation_zx' in result
    assert 'rotation_zy' in result

    # Check shapes
    assert result['rotation_xy'].shape == (21,)
    assert result['rotation_zx'].shape == (21,)
    assert result['rotation_zy'].shape == (21,)


@pytest.mark.slow
def test_real_data_projection_types(single_channel_file, temp_output_dir):
    """Test different projection types with real data."""
    projection_types = ['average', 'max']  # Test subset for speed

    for proj_type in projection_types:
        result = register_image_from_file(
            single_channel_file,
            axis_order="TZYX",
            ref_channel=0,
            output_dir=temp_output_dir / f"proj_{proj_type}",
            correct_xy=True,
            correct_z=False,
            correct_rotation=False,
            projection_type=proj_type,
            return_drifts=True,
        )

        # Should succeed and detect some drift
        assert 'registered_image' in result
        assert result['xy_drift'].shape == (21, 2)


@pytest.mark.slow
def test_real_data_reference_modes(single_channel_file, temp_output_dir):
    """Test different reference modes with real data."""
    for ref_mode in ['relative', 'first_frame']:
        result = register_image_from_file(
            single_channel_file,
            axis_order="TZYX",
            ref_channel=0,
            output_dir=temp_output_dir / f"ref_{ref_mode}",
            correct_xy=True,
            correct_z=False,
            correct_rotation=False,
            reference_mode=ref_mode,
            return_drifts=True,
        )

        # Should succeed
        assert 'registered_image' in result
        assert result['xy_drift'].shape == (21, 2)

        if ref_mode == 'first_frame':
            # First frame should have zero drift
            assert np.allclose(result['xy_drift'][0], 0)
