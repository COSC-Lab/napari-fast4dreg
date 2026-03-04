# Fast4DReg API Quick Reference

## Installation

### Full Installation (with napari GUI plugin)

```bash
pip install napari-fast4dreg
```

### API-Only Installation (lightweight, no GUI dependencies)

```bash
pip install napari-fast4dreg[api-only]
```

Use the API-only installation if you only need programmatic access and want to avoid installing napari and its dependencies.

## Three Ways to Use Fast4DReg

### 1. napari GUI (Interactive)

```python
import napari
viewer = napari.Viewer()
# Load image in viewer, then use Plugins → napari-fast4dreg → Fast4DReg
```

### 2. Programmatic API (In Your Script)

```python
from napari_fast4dreg import register_image

# With numpy/dask array - CTZYX format
result = register_image(
    image,  
    axis_order="CTZYX",
    ref_channel=0,
    output_dir="./results"
)

# Or TZYX format (4D, single channel)
result = register_image(
    image,  
    axis_order="TZYX",
    ref_channel=0,
    output_dir="./results"
)

registered = result['registered_image']
xy_drift = result['xy_drift']
```

### 3. Load from File

```python
from napari_fast4dreg import register_image_from_file

# TIFF file (ImageJ format)
result = register_image_from_file(
    "my_image.tif",
    axis_order="TZCYX",  # ImageJ format
    ref_channel=1,
    output_dir="./results"
)

# NumPy binary file
result = register_image_from_file(
    "my_image.npy",
    axis_order="CZYX",
    ref_channel=0,
    output_dir="./results"
)

# Zarr file (chunked)
result = register_image_from_file(
    "my_image.zarr",
    axis_order="CTZYX",
    ref_channel=0,
    output_dir="./results"
)

# Output is always same shape as input shape
registered = result['registered_image']  
xy_drift = result['xy_drift']
```

## Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | ndarray/dask | required | Image in specified axis order |
| `axis_order` | str | "CTZYX" | Axis order: CTZYX, TZCYX, TZYX, ZYX, CZYX, CYX, TYX, or YX. Missing axes auto-added |
| `ref_channel` | int/str/list/tuple | 0 | Reference channel(s): int (e.g., `0`), list (e.g., `[0,1,5]`), comma-separated string (e.g., `"0,1,5"`), space-separated (e.g., `"0 1 5"`) |
| `output_dir` | str/Path | "./fast4dreg_output" | Output directory |
| `correct_xy` | bool | True | Apply XY drift correction |
| `correct_z` | bool | True | Apply Z drift correction |
| `correct_rotation` | bool | True | Apply 3D rotation correction (sequential: XY→ZX→ZY) |
| `crop_output` | bool | False | Crop invalid regions |
| `projection_type` | str | 'average' | 'average', 'max', 'median', 'min' |
| `reference_mode` | str | 'relative' | 'relative' or 'first_frame' |
| `normalize_channels` | bool | False | Normalize multi-channel reference |
| `progress_callback` | callable | None | Function for progress updates |
| `return_drifts` | bool | True | Include drift data in result |
| `keep_temp_files` | bool | False | Keep temporary Zarr stores |

## Return Value

Dictionary containing:
```python
{
    'registered_image': dask.array.Array,  # Registered image (dask array, lazy-loaded)
                                          # For register_image: same axis order as input
                                          # For register_image_from_file: always CTZYX
    'xy_drift': np.ndarray,          # XY drift values (T, 2)
    'z_drift': np.ndarray,           # Z drift values (T, 2)
    'rotation_xy': np.ndarray,       # XY rotation angles
    'rotation_zx': np.ndarray,       # ZX rotation angles
    'rotation_zy': np.ndarray,       # ZY rotation angles
    'output_path': Path              # Path to saved Zarr
}
```

### Working with the Registered Image

The `registered_image` is a **dask array**, meaning it's lazy-loaded from Zarr storage:

```python
result = register_image(image, ...}
registered = result['registered_image']  # dask.array, not in RAM

# Convert to numpy if needed (loads into memory, make sure it fits)
registered_np = registered.compute()

# Or work with it directly for out-of-memory processing
mip_projection = registered.max(axis = 2).compute()

# Or save chunks back to disk
import dask.array as da
da.to_zarr(registered, "path/to/output.zarr")
```

## Supported Axis Orders

The `axis_order` parameter accepts flexible axis specifications:

| Order | Shape | Description |
|-------|-------|-------------|
| `CTZYX` | (C, T, Z, Y, X) | Standard 5D format |
| `TZYX` | (T, Z, Y, X) | 4D, single channel |
| `ZCYX` | (Z, C, Y, X) | 4D, single timepoint |
| `CZYX` | (C, Z, Y, X) | 4D, single timepoint |
| `ZYX` | (Z, Y, X) | 3D single timepoint + channel |
| `TYX` | (T, Y, X) | Time series 2D |
| `CYX` | (C, Y, X) | Multi-channel 2D image (included for pipeline savety) |
| `YX` | (Y, X) | Single 2D image (included for pipeline savety) | 

Missing dimensions are automatically inserted as singletons during processing and removed in the output.

## Examples

### Basic Usage

```python
from napari_fast4dreg import register_image
import numpy as np

image = np.load("my_image.npy")  # TZYX format (4D)
result = register_image(
    image, 
    axis_order="TZYX",
    ref_channel=0
)
registered = result['registered_image']
```

### With Different Axis Orders

```python
# ImageJ format (TZCYX)
result = register_image(
    image, 
    axis_order="TZCYX",
    ref_channel=0
)

# Simple 3D volume (ZYX)
result = register_image(
    image,
    axis_order="ZYX",
    ref_channel=0
)

# 2D time series (TYX)
result = register_image(
    image,
    axis_order="TYX",
    ref_channel=0
)
```

### With Progress Tracking

```python
def show_progress(msg):
    print(f"[{datetime.now()}] {msg}")

result = register_image(
    image,
    ref_channel=0,
    progress_callback=show_progress
)
```

### Only XY Correction (Fast)

```python
result = register_image(
    image,
    ref_channel=0,
    correct_xy=True,
    correct_z=False,
    correct_rotation=False
)
```

### Multi-Channel Reference

```python
# Multiple ways to specify multiple reference channels:

# Option 1: List of integers
result = register_image(
    image,
    ref_channel=[0, 3, 5],  # List of channel indices
    normalize_channels=True,
    projection_type='max'
)

# Option 2: Tuple of integers
result = register_image(
    image,
    ref_channel=(0, 3, 5),  # Tuple - same as list
    normalize_channels=True,
    projection_type='max'
)

# Option 3: Comma-separated string
result = register_image(
    image,
    ref_channel="0,3,5",  # Comma-separated string
    normalize_channels=True,
    projection_type='max'
)

# Option 4: Space-separated string
result = register_image(
    image,
    ref_channel="0 3 5",  # Space-separated string
    normalize_channels=True,
    projection_type='max'
)
```

All formats above produce the same result: channels 0, 3, and 5 are summed together for drift detection.

### Integration into Workflow

```python
from napari_fast4dreg import register_image
import numpy as np

# Step 1: Your preprocessing
preprocessed = your_preprocessing_function(raw_data)

# Step 2: Registration
result = register_image(
    preprocessed,
    ref_channel=1,
    output_dir="./registration_output",
    return_drifts=True
)

# Step 3: Your postprocessing
final = your_postprocessing_function(result['registered_image'])

# Step 4: Analyze drift
xy_drift = result['xy_drift']
max_drift = np.sqrt(xy_drift[:, 0]**2 + xy_drift[:, 1]**2).max()
print(f"Maximum drift: {max_drift:.2f} pixels")
```

### From TIFF File (ImageJ Format)

```python
from napari_fast4dreg import register_image_from_file

# ImageJ format (TZCYX) - Output will be TZCYX
result = register_image_from_file(
    "timelapse.tif",
    axis_order="TZCYX",  # Time, Z, Channel, Y, X (default for TIFF)
    ref_channel=1,
    output_dir="./results"
)
registered = result['registered_image']  # Shape: (T, Z, C, Y, X)

# Single channel TIFF (TZYX) - Output will be TZYX  
result = register_image_from_file(
    "timelapse.tif",
    axis_order="TZYX",   # Time, Z, Y, X
    ref_channel=0,
    output_dir="./results"
)
registered = result['registered_image']  # Shape: (T, Z, Y, X)

# NumPy array file (.npy)
result = register_image_from_file(
    "image_stack.npy",
    axis_order="CZYX",   # Channels, Z, Y, X
    ref_channel="0,1",
    output_dir="./results"
)
registered = result['registered_image']  # Shape: (C, Z, Y, X)
```

### Batch Processing

```python
from napari_fast4dreg import register_image_from_file
from pathlib import Path

input_dir = Path("./raw_data")
output_dir = Path("./registered")

for tif_file in input_dir.glob("*.tif"):
    print(f"Processing {tif_file.name}...")
    
    result = register_image_from_file(
        tif_file,
        axis_order="TZCYX",
        ref_channel=0,
        output_dir=output_dir / tif_file.stem,
        progress_callback=lambda msg: print(f"  {msg}")
    )
    
    print(f"  Saved to: {result['output_path']}\n")
```

## Axis Order Formats

Fast4DReg processes images in **CTZYX** format internally, but input and output formats are flexible.

### Input/Output Axis Orders
- **register_image()**: Input in format X, output in format X (same as input)
- **register_image_from_file()**: Input in format X, output in format X (same as input file)

### Supported Formats
- `CTZYX` - Standard 5D (Channels, Time, Z, Y, X)
- `TZCYX` - ImageJ/Fiji format (Time, Z, Channels, Y, X) [DEFAULT for register_image_from_file]
- `TZYX` - Single channel time series (Time, Z, Y, X)
- `CZYX` - Multi-channel single timepoint (Channels, Z, Y, X)
- `ZYX` - Single timepoint single channel volume (Z, Y, X)
- `CYX` - Multi-channel 2D image (Channels, Y, X)
- `TYX` - 2D time series (Time, Y, X)
- `YX` - Single 2D image

Missing dimensions (C, T, Z) are automatically inserted as singletons during processing.

### Example: Axis Order Preservation

```python
from napari_fast4dreg import register_image, register_image_from_file
import numpy as np

# register_image preserves input format
image_tzyx = np.load("data.npy")  # Shape: (T, Z, Y, X)
result = register_image(image_tzyx, axis_order="TZYX")
output = result['registered_image']  # Still TZYX: (T, Z, Y, X)

# register_image_from_file preserves file format
# File is TZCYX, output is TZCYX
result = register_image_from_file("data.tif", axis_order="TZCYX")
output = result['registered_image']  # TZCYX: (T, Z, C, Y, X)

# Custom format also preserved
result = register_image_from_file("data.npy", axis_order="CZYX")
output = result['registered_image']  # CZYX: (C, Z, Y, X)
```

## Output Files

When registration completes, you'll find:
- `registered.zarr/` - Final registered image (Zarr format, dask-accessible)
   - Format matches your input's axis_order
   - For TZCYX input: output is TZCYX
   - For CZYX input: output is CZYX
   - etc.
- `tmp_data_*.zarr/` - Temporary stores (deleted unless `keep_temp_files=True`)

Load and work with the output:
```python
# From the result dictionary
result = register_image_from_file("data.tif", axis_order="TZCYX")
registered = result['registered_image']  # dask.array in TZCYX format

# Or load manually from disk
import dask.array as da
registered = da.from_zarr("./results/registered.zarr")  # Format matches input

# Convert to numpy for analysis
registered_np = registered.compute()

# Or process chunks without loading all into RAM
mean = registered.mean().compute()
max_val = registered.max().compute()

# View in napari (napari accepts dask arrays directly)
import napari
viewer = napari.view_image(registered)
```

## Tips

1. **Large datasets**: Use dask arrays instead of loading to RAM
2. **Speed**: Disable rotation correction if not needed
3. **Accuracy**: Use max projection for sparse bright features
4. **Multi-channel**: Enable normalization if intensities differ greatly
5. **Storage**: Clean up temp files with `keep_temp_files=False` (default)
6. **Rotation**: Rotations are estimated and applied sequentially (XY plane first, then ZX, then ZY) on the already-corrected data for improved accuracy
7. **GPU Acceleration**: Automatic GPU detection and acceleration. Prefers NVIDIA over Intel GPUs

## GPU Acceleration (Optional)

Fast4DReg **automatically detects and enables GPU acceleration** when available using [pyclesperanto](https://github.com/clEsperanto/pyclesperanto) for all transformation operations (translations and rotations).

### Installation

```bash
# Install pyclesperanto for GPU support
pip install pyclesperanto
```

### Automatic Detection

When you import napari-fast4dreg, it will:
1. **Automatically detect** available GPUs
2. **Prefer NVIDIA** GPUs over Intel GPUs
3. **Enable GPU acceleration** if a suitable GPU is found
4. **Display GPU info** in the napari widget
5. **Fall back to CPU** if GPU memory is insufficient for a transform

```python
from napari_fast4dreg import register_image, get_gpu_info

# Check which backend is being used
print(get_gpu_info())  # e.g., "GPU (NVIDIA GeForce RTX 3080)" or "CPU (scipy)"

# Run registration - GPU will be used automatically if available
result = register_image(
    image,
    ref_channel=0,
    output_dir="./results"
)
```

### Manual Control (Optional)

You can manually enable/disable GPU acceleration:

```python
from napari_fast4dreg import set_gpu_acceleration, get_gpu_info

# Manually enable GPU (if available)
success = set_gpu_acceleration(True)
print(f"GPU enabled: {success}")
print(f"Backend: {get_gpu_info()}")

# Disable GPU acceleration (use CPU)
set_gpu_acceleration(False)
print(f"Backend: {get_gpu_info()}")  # "CPU (scipy)"
```

### GPU Priority

When multiple GPUs are available, Fast4DReg selects in this order:
1. **NVIDIA** (GeForce, RTX, GTX, Quadro, Tesla)
2. **Intel** (Iris, HD Graphics)
3. **Other** OpenCL-compatible devices

**Performance**: GPU acceleration can provide 5-10x speedup for transformation steps, especially beneficial for:
- Large images (>1GB)
- High timepoint counts (>50 frames)
- 3D rotation corrections

**Requirements**: 
- OpenCL-compatible GPU (NVIDIA, AMD, or Intel)
- pyclesperanto installed
- Sufficient GPU memory for image data

**Notes**:
- If OpenCL only reports a CPU device, install the NVIDIA OpenCL ICD for your driver.
- If a transform runs out of VRAM, Fast4DReg switches to CPU automatically and continues.

**Note**: The napari widget displays the current processing backend (GPU device name or CPU) in the "Processing Backend" field.

## Need Help?

- Full documentation: README.md
- Example scripts: `examples_api_usage.py`
- Issues: https://github.com/COSC-Lab/napari-fast4dreg/issues
- Contact: marcel.issler@kuleuven.be
