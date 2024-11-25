# napari-fast4dreg

[![License BSD-3](https://img.shields.io/pypi/l/napari-fast4dreg.svg?color=green)](https://github.com/Macl-I/napari-fast4dreg/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-fast4dreg.svg?color=green)](https://pypi.org/project/napari-fast4dreg)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-fast4dreg.svg?color=green)](https://python.org)
[![tests](https://github.com/Macl-I/napari-fast4dreg/workflows/tests/badge.svg)](https://github.com/Macl-I/napari-fast4dreg/actions)
[![codecov](https://codecov.io/gh/Macl-I/napari-fast4dreg/branch/main/graph/badge.svg)](https://codecov.io/gh/Macl-I/napari-fast4dreg)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-fast4dreg)](https://napari-hub.org/plugins/napari-fast4dreg)

Dask empowered multi-dimensional, registration for volumetric measurements.
This is a python port of the original Fast4DReg Fiji Plugin, with added rotation correction in lateral direction and support for out of memory processing.
The original paper can be found here:
https://journals.biologists.com/jcs/article/136/4/jcs260728/287682/Fast4DReg-fast-registration-of-4D-microscopy

----------------------------------


<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->
## Suggested Changes
- [ ] add environment setup and full install of miniconda etc to install section
- [x] implement zarr storage instead of npy stacks - Using alternating files to store intermediate data (consider moving to one file)
- [x] test alternative affine transform functions, such as simpple-ITK or openCV - settled for dask_image (gpu supported), consider looking up enforcing gpu usage
- [x] consinder adding cuda support (nvidia gpu only) - replaced by dask_image
## Installation

You can install `napari-fast4dreg` via [pip]:

    pip install napari-fast4dreg

## Usage 

It's easy! 
1) Just drag and drop your image, or the test image from this repository, into napari and open it normally. 
Don't worry if your file is big, napari already internally uses dask to open even the biggest images (although it might hurt the performance).
2) Open the napari-fast4dreg plugin from the plugin menu.
3) In the image row, make sure your image is selected in the image drop down menu.
4) In the axes row, choose the structure of your input image. If your axis orientation is correct in ImageJ choose the standard TZCYX (ImageJ) orientation. If you are using python to process the image you probabbly are using the alternatively availabe CTZYX orientation. In this case just select CTZYX in the drop down menu instead.
5) Select the reference channel used for the registration. The drift will be determined for this reference channel and applied to all other channels. Counting begins by 0. In case for the test image we select the nuclear signal in channel 1.
6) Select the corrections that you want to apply on your image. Note that the crop function reduces only in xy, according to the previously determined drift. (e.g. drift = -5 in x --> drop 5 pixels from the left hand side of the registered stack.)
7) Wait for output (this may take a while, so go and get a coffe or tea).
8) Enjoy your registered image.


## Example Outcome
The output will consist of the following (if chosen): 
- registered.tif: The registered file, output of this image registration pipeline.
- tmp_data: This folder was used for temporary data saving and stores at the end the registered image in a chunked manner (can be deleted or dragged into napari for a greater data versatility)
- drifts.csv: csv table, home to the drift of all corrected variables, if you prefer your own plotting style, here is where you find the pure drift table.
- XY-Drift.svg: Vector based graphic, visualising the drift in lateral direction. The svg format can be opened by your web browser or directly imported to powerpoint. Key advantage of .svg instead of .png: You can resize any way you like without loss of image quality.
- Z-Drift.svg: Vector based graphic, visualising the drift in axial direction.
- Rotation-Drift.svg: Showing rotation correction of the image in lateral direction.
  
![3D_MIP_registration](./media/3D_registration.gif)
![3D_plane](./media/3D_plane_registration.gif)
![XY-Drift](./media/XY-Drift.svg)
![Z-Drift](./media/Z-Drift.svg)
![Rotation-Drift](./media/Rotation-Drift.svg)

## Contributing

Contributions are very welcome. Just send me an E-mail: marcel.issler@kuleuven.be or directly submit a pull request.

## Credit 
This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

## License

Distributed under the terms of the [BSD-3] license,
"napari-fast4dreg" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
