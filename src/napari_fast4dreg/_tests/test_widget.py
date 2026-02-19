import numpy as np
import napari 
import os 
from pathlib import Path

from napari_fast4dreg._widget import Fast4DReg_widget 


def test_Fast4DReg_widget():
    """Test that the Fast4DReg widget can be instantiated."""
    # Create viewer (required for napari widgets)
    viewer = napari.Viewer()
    
    # Create widget
    widget = Fast4DReg_widget()
    
    # Check that widget has expected attributes
    assert hasattr(widget, 'image_layer')
    assert hasattr(widget, 'axes')
    assert hasattr(widget, 'ref_channel')
    assert hasattr(widget, 'normalize_channels')
    assert hasattr(widget, 'projection_type')
    assert hasattr(widget, 'reference_mode')
    assert hasattr(widget, 'output_path')
    assert hasattr(widget, 'multichannel_mode')
    assert hasattr(widget, 'correct_xy')
    assert hasattr(widget, 'correct_z')
    assert hasattr(widget, 'correct_rotation')
    assert hasattr(widget, 'crop_output')
    assert hasattr(widget, 'export_data')
    assert hasattr(widget, 'progress_bar')
    assert hasattr(widget, 'run_btn')
    
    # Check default values
    assert widget.correct_xy.value == True
    assert widget.correct_z.value == True
    assert widget.correct_rotation.value == True
    assert widget.crop_output.value == False  # Updated default
    assert widget.projection_type.value == "Average"
    assert widget.reference_mode.value == "Relative"
    
    # Close viewer
    viewer.close()
