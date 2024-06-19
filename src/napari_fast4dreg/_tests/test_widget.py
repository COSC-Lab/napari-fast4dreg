import numpy as np
import tifffile 

from napari_fast4dreg._widget import Fast4DReg_widget

def test_Fast4DReg_widget():
    # working path is napari-fast4dreg/napari-fast4dreg/
    # 
    
    test_image = tifffile.imread('example_files/xtitched_organoid_timelapse_ch0_cytosol_ch1_nuclei.tif')
    reference_registered_image = tifffile.imread('example_files/registered.tif')
    out_folder = 'example_files/'
    resgistered_image = Fast4DReg_widget(image = test_image,
                                        axes = 1, 
                                        output_path = out_folder, 
                                        ref_channel = r"1", 
                                        correct_xy = True,
                                        correct_z = True,
                                        correct_center_rotation = True,
                                        crop_output = True, 
                                        export_csv = True )
    
    assert resgistered_image == reference_registered_image