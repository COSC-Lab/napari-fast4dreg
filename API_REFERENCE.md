# Fast4DReg API Quick Reference

## Installation

```bash
pip install napari-fast4dreg
```

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

# With numpy/dask array
result = register_image(
    image,  # CTZYX format
    ref_channel=0,
    output_dir="./results"
)

registered = result['registered_image']
xy_drift = result['xy_drift']
```

### 3. Load from File

```python
from napari_fast4dreg import register_image_from_file

result = register_image_from_file(
    "my_image.tif",
    axis_order="TZCYX",  # ImageJ format
    ref_channel=1,
    output_dir="./results"
)
```

## Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | ndarray/dask | required | Image in CTZYX format |
| `ref_channel` | int/str | 0 | Reference channel(s): int (e.g., `0`), comma-separated (e.g., `"0,1"`), or space-separated (e.g., `"0 1"`) |
| `output_dir` | str/Path | "./fast4dreg_output" | Output directory |
| `correct_xy` | bool | True | Apply XY drift correction |
| `correct_z` | bool | True | Apply Z drift correction |
| `correct_rotation` | bool | True | Apply 3D rotation correction |
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
    'registered_image': np.ndarray,  # Registered image (CTZYX)
    'xy_drift': np.ndarray,          # XY drift values
    'z_drift': np.ndarray,           # Z drift values
    'rotation_xy': np.ndarray,       # XY rotation angles
    'rotation_zx': np.ndarray,       # ZX rotation angles
    'rotation_zy': np.ndarray,       # ZY rotation angles
    'output_path': Path              # Path to saved Zarr
}
```

## Examples

### Basic Usage

```python
from napari_fast4dreg import register_image
import numpy as np

image = np.load("my_image.npy")  # CTZYX format
result = register_image(image, ref_channel=0)
registered = result['registered_image']
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
# Use multiple channels - comma or space-separated
result = register_image(
    image,
    ref_channel="0,3,5",  # Use channels 0, 3, and 5 (comma-separated)
    # OR: ref_channel="0 3 5"  # Space-separated also works
    normalize_channels=True,
    projection_type='max'
)
```

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

result = register_image_from_file(
    "timelapse.tif",
    axis_order="TZCYX",  # ImageJ: Time, Z, Channel, Y, X
    ref_channel=1,
    output_dir="./results"
)
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

Fast4DReg expects **CTZYX** format (Channels, Time, Z, Y, X).

Common conversions:
- **ImageJ/Fiji** (`TZCYX`): Use `axis_order="TZCYX"` in `register_image_from_file()`
- **Single channel** (`TZYX`): Use `axis_order="TZYX"` - channel dimension is added automatically
- **Already CTZYX**: Use `axis_order="CTZYX"` or pass directly to `register_image()`

Manual conversion:
```python
import numpy as np

# TZCYX → CTZYX
image_ctzyx = np.moveaxis(image_tzcyx, [0, 1, 2], [1, 2, 0])

# TZYX → CTZYX (add channel dim)
image_ctzyx = image_tzyx[:, np.newaxis, ...]
```

## Output Files

When registration completes, you'll find:
- `registered.zarr/` - Final registered image (Zarr format)
- `tmp_data_*.zarr/` - Temporary stores (deleted unless `keep_temp_files=True`)

Load the output in napari or Python:
```python
# In napari
viewer.open("./results/registered.zarr")

# In Python
import dask.array as da
import zarr
registered = da.from_zarr("./results/registered.zarr")
```

## Tips

1. **Large datasets**: Use dask arrays instead of loading to RAM
2. **Speed**: Disable rotation correction if not needed
3. **Accuracy**: Use max projection for sparse bright features
4. **Multi-channel**: Enable normalization if intensities differ greatly
5. **Storage**: Clean up temp files with `keep_temp_files=False` (default)

## Need Help?

- Full documentation: README.md
- Example scripts: `examples_api_usage.py`
- Issues: https://github.com/COSC-Lab/napari-fast4dreg/issues
- Contact: marcel.issler@kuleuven.be
