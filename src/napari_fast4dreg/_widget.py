"""
This module contains four napari widgets declared in
different ways:

- a pure Python function flagged with `autogenerate: true`
    in the plugin manifest. Type annotations are used by
    magicgui to generate widgets for each parameter. Best
    suited for simple processing tasks - usually taking
    in and/or returning a layer.
- a `magic_factory` decorated function. The `magic_factory`
    decorator allows us to customize aspects of the resulting
    GUI, including the widgets associated with each parameter.
    Best used when you have a very simple processing task,
    but want some control over the autogenerated widgets. If you
    find yourself needing to define lots of nested functions to achieve
    your functionality, maybe look at the `Container` widget!
- a `magicgui.widgets.Container` subclass. This provides lots
    of flexibility and customization options while still supporting
    `magicgui` widgets and convenience methods for creating widgets
    from type annotations. If you want to customize your widgets and
    connect callbacks, this is the best widget option for you.
- a `QWidget` subclass. This provides maximal flexibility but requires
    full specification of widget layouts, callbacks, events, etc.

References:
- Widget specification: https://napari.org/stable/plugins/guides.html?#widgets
- magicgui docs: https://pyapp-kit.github.io/magicgui/

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
from magicgui import magicgui
from superqt.utils import thread_worker
import time
import os
import numpy as np
import dask.array as da
import pandas as pd
import tifffile 
from enum import Enum
from ._fast4Dreg_functions import get_xy_drift, apply_xy_drift
from ._fast4Dreg_functions import get_z_drift, apply_z_drift
from ._fast4Dreg_functions import get_rotation, apply_alpha_drift
from ._fast4Dreg_functions import crop_data
from ._fast4Dreg_functions import read_tmp_data, write_tmp_data_to_disk

from magicgui.tqdm import tqdm
from dask import delayed
import napari 
import shutil

if TYPE_CHECKING:
    import napari


# Uses the `autogenerate: true` flag in the plugin manifest
# to indicate it should be wrapped as a magicgui to autogenerate
# a widget.

class Axes(Enum):
    """str for various media and their refractive indices."""

    CTZYX = 0
    TZCYX_ImageJ = 1



def Fast4DReg_widget(
    image: "napari.types.ImageData",
    axes = Axes.TZCYX_ImageJ, 
    output_path = r"E:\Data_Leuven\Fast4DReg_Dask\plugin_output_trial", 
    ref_channel = r"1", 
    correct_xy = True,
    correct_z = True,
    correct_center_rotation = True,
    crop_output = True, 
    export_csv = True 
    ):

    # Code for proper progress bar. Propbably requires more intense rewriting tho
    # n_steps = int(correct_xy+correct_z + correct_center_rotation + crop_output + export_csv) 
    # n_steps = n_steps + 2 # import and export need to be added too 
          
    # start timer
    with tqdm() as pbar:
        
        @thread_worker(connect={"finished": lambda: pbar.progressbar.hide()})
        def run_pipeline(image, 
                            axes, 
                            output_path, 
                            ref_channel, 
                            correct_xy, 
                            correct_z, 
                            correct_center_rotation, 
                            crop_output, 
                            export_csv
                            ): 
            
            # start timer
            start_time = time.time()

            # rephrase variables 
            image = da.asarray(image)
            
            # check for multichannel nature 
            if len(image.shape) == 4: 
                # --> new image shape is now TZYX for both cases
                # extend image to fit processing format of CTZYX
                data = da.asarray([image])
                # set ref_channel to 0, since its the only existing channel
                # this effectively overwrites user input, making it irrelevant 
                ref_channel = 0
                print('Single-Channel Image detected \nNew Shape: ', image.shape)

            else: 
                # process as multichannel
                ## reorder image if necessary: 
                if axes.value == 0: 
                    data = image
                if axes.value == 1: 
                    # move channel domain to front
                    image = image.swapaxes(0,2)
                    # swap time and z 
                    data = image.swapaxes(1,2)
                    print(np.shape(data))
            
            print('Reshaped order of the imput image (supposed to be CTZYX): {}'.format(np.shape(image))) 
            
            output_dir = output_path
            os.chdir(output_dir)    
            
            # tmp_file for read/write
            tmp_path = str(output_dir + '//tmp_data//')
                        
            # reference channel is channel 1, where the nuclei are imaged
            ref_channel = int(ref_channel)
            
            # if ref_channel index out of range take the last channel of the image
            if ref_channel > len(data[0]): 
                ref_channel = len(data[0])
            
            # read in raw data as dask array
            #new_shape = (np.shape(data)[0],1,np.shape(data)[-3],np.shape(data)[-2],np.shape(data)[-1])
            data = data.rechunk('auto')
            new_shape = data.chunksize
                                
            # write data to tmp_file
            data = write_tmp_data_to_disk(tmp_path, data, new_shape)
            print('Imge imported')
            yield pbar.update(1)
            # Run the method 
            if correct_xy == True: 
                xy_drift = get_xy_drift(data, ref_channel)
                tmp_data = apply_xy_drift(data, xy_drift)
                # save intermediate results to temporary npy file
                tmp_data = write_tmp_data_to_disk(tmp_path, tmp_data, new_shape)
                yield pbar.update(1)
            else: 
                tmp_data = data
                xy_drift = np.asarray([[0,0]])
            
            if correct_z == True: 
                # Correct z-drift
                z_drift = get_z_drift(data, ref_channel)
                tmp_data = apply_z_drift(tmp_data, z_drift)

                # save intermediate result
                tmp_data = write_tmp_data_to_disk(tmp_path, tmp_data, new_shape)

                yield pbar.update(1)
                
            else: 
                z_drift = np.asarray([[0,0]])
            
            if  crop_output == True: 
                # Crop, according to drift
                tmp_data = crop_data(tmp_data, xy_drift, z_drift)
                new_shape = (np.shape(tmp_data)[0],1,np.shape(tmp_data)[-3],np.shape(tmp_data)[-2],np.shape(tmp_data)[-1])
                
                # save intermediate result
                crop_path = output_dir +"//cropped_tmp_data"
                da.to_npy_stack(crop_path,tmp_data, axis = 1)
                del tmp_data
                shutil.rmtree(tmp_path)
                shutil.move(crop_path, tmp_path)
                tmp_data = read_tmp_data(tmp_path, new_shape)

                yield pbar.update(1)
            
            if correct_center_rotation == True: 
                # Correct Rotation 
                alpha = get_rotation(tmp_data, ref_channel)
                tmp_data = apply_alpha_drift(tmp_data, alpha)
                
                # save intermediate result
                tmp_data = write_tmp_data_to_disk(tmp_path, tmp_data, new_shape)

                yield pbar.update(1)
                
            else: 
                alpha = [0]
            
            if export_csv == True:
                # Export .csv
                print("Export drifts to csv.")
                x = pd.DataFrame({'x-drift': xy_drift[:,0]})
                y = pd.DataFrame({'y-drift': xy_drift[:,1]})
                z = pd.DataFrame({'z-drift': z_drift[:,0]})
                r = pd.DataFrame({'rotation': alpha})
                df = pd.concat([x,y,z,r], axis=1)
                df = df.fillna(0)
                df.to_csv("drifts.csv")
                
                yield pbar.update(1)

            # move axis back to ImageJ format if necessary
            if len(image.shape)!=4: 
                if axes.value == 1: 
                    tmp_data = tmp_data.swapaxes(0,1)
                    tmp_data = tmp_data.swapaxes(1,2)
                    print(np.shape(tmp_data))            
                yield pbar.update(1)
            else: 
                yield pbar.update(1)
                
            
             # write results to tif
            export_path = output_dir + "/registered.tif"
            tifffile.imwrite(export_path, tmp_data, ome=True)
            
            # print report
            print('Rigid Fast4D Registration complete.')
            print("--- %s seconds ---" % (time.time() - start_time))        
            
            return tmp_data
        # grab viewer
        viewer = napari.current_viewer()
        
        # initiate worker
        worker = run_pipeline(image, 
                         axes, 
                         output_path, 
                         ref_channel, 
                         correct_xy, 
                         correct_z, 
                         correct_center_rotation,
                         crop_output,
                         export_csv
                         )
        
        # visualise worker once its done
        worker.returned.connect(viewer.add_image)
