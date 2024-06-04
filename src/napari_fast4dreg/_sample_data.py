"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations

import numpy
import tifffile 
import io
import requests


def make_sample_data():
    """Load image"""
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image    
    # Check that request succeeded
    return tifffile.imread(r'C:\Users\u0166007\Documents\Napari-Stuff\Plugin_Sandbox\napari-fast4dreg\src\napari_fast4dreg\test_data\xtitched_organoid_timelapse_ch0_cytosol_ch1_nuclei.tif') 

