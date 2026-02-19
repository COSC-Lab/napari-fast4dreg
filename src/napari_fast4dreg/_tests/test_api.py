"""
Tests for the programmatic API (register_image, register_image_from_file).
"""
import numpy as np
import pytest
import tempfile
from pathlib import Path
import shutil

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
    """Test that invalid image shape raises error."""
    # 4D image (not 5D)
    invalid_image = np.random.randint(0, 255, (3, 4, 32, 32), dtype=np.uint8)
    
    with pytest.raises(ValueError, match="Image must be 5D"):
        register_image(
            invalid_image,
            ref_channel=0,
            output_dir="./test_output"
        )


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
