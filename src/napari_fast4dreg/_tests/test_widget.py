import numpy as np
import tifffile 
import napari 

from napari_fast4dreg._widget import Fast4DReg_widget 

def test_Fast4DReg_widget():
    # working path is napari-fast4dreg/napari-fast4dreg/
    # 
    viewer = napari.Viewer()
    test_image = tifffile.imread('example_files/xtitched_organoid_timelapse_ch0_cytosol_ch1_nuclei.tif')
    reference_registered_image = tifffile.imread('example_files/registered.tif')
    viewer.add_image(reference_registered_image)

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
    
    # get layer data 
    layers = viewer.layers()
    
    assert layers[0].data == layers[-1].data