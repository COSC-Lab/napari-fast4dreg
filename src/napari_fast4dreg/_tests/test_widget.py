import numpy as np
import tifffile 

from napari_fast4dreg._widget import (
    Fast4DReg_widget
)

def test_Fast4DReg_widget():
    test_image = tifffile.imread('test_input.tif')
    reference_registered_image = tifffile.imread('registered.tif')
    resgistered_image = Fast4DReg_widget(image: test_image,
                                        axes = Axes.TZCYX_ImageJ, 
                                        output_path = r"E:\Data_Leuven\Fast4DReg_Dask\plugin_output_trial", 
                                        ref_channel = r"1", 
                                        correct_xy = True,
                                        correct_z = True,
                                        correct_center_rotation = True,
                                        crop_output = True, 
                                        export_csv = True )
    assert resgistered_image == reference_registered_image