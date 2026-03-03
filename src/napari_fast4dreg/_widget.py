"""
napari-fast4dreg widget for 4D image registration.
npe2 compatible widget using magicgui with progress bar.
"""
import os
import shutil
import time
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import dask.array as da
import napari.layers
import numpy as np
import pandas as pd
from magicgui.widgets import Container, ProgressBar, PushButton, create_widget
from napari.qt.threading import thread_worker

from ._axis_utils import (
    convert_to_ctzyx,
    parse_axis_order,
    revert_to_original_axis_order,
)
from ._fast4Dreg_functions import (
    apply_alpha_drift,
    apply_beta_drift,
    apply_gamma_drift,
    apply_xy_drift,
    apply_z_drift,
    crop_data,
    get_gpu_info,
    get_rotation_alpha,
    get_rotation_beta,
    get_rotation_gamma,
    get_xy_drift,
    get_z_drift,
    read_tmp_data,
    set_gpu_acceleration,
    write_tmp_data_to_disk,
)

if TYPE_CHECKING:
    import napari


class Axes(Enum):
    """Axis order options for image data."""
    CTZYX = 0
    TZCYX_ImageJ = 1
    TZYX = 2
    ZCYX = 3
    CZYX = 4
    ZYX = 5


# Set default output path to example_files folder
_DEFAULT_OUTPUT_PATH = str(Path(__file__).parent.parent.parent / "example_files")



# Set default output path to example_files folder
_DEFAULT_OUTPUT_PATH = str(Path(__file__).parent.parent.parent / "example_files")


class Fast4DRegWidget(Container):
    """Unified Fast4DReg registration widget with progress bar."""

    def __init__(self, napari_viewer: "napari.Viewer" = None):
        super().__init__(labels=False)

        print(f"DEBUG: Fast4DRegWidget.__init__ called with napari_viewer = {napari_viewer}")
        print(f"DEBUG: napari_viewer type = {type(napari_viewer)}")
        self.viewer = napari_viewer
        print(f"DEBUG: self.viewer set to {self.viewer}")

        # Create widgets
        self.image_layer = create_widget(
            annotation=napari.layers.Image,
            label="Image Layer",
            name="image_layer"
        )

        self.axes = create_widget(
            annotation=str,
            label="Axis Order",
            name="axes",
            value="CTZYX"
        )
        self.axes.tooltip = "Specify axis order as string (e.g., CTZYX, TZCYX, TZYX, ZYX, etc.)"

        self.ref_channel = create_widget(
            annotation=str,
            label="Reference Channel(s)",
            name="ref_channel",
            value="0"
        )

        self.normalize_channels = create_widget(
            annotation=bool,
            label="Normalize Channels (for multi-channel ref)",
            name="normalize_channels",
            value=False
        )

        self.projection_type = create_widget(
            annotation=str,
            label="Projection Type",
            name="projection_type",
            widget_type="ComboBox",
            options={"choices": ["Average", "Max", "Median", "Min"]}
        )
        self.projection_type.value = "Average"

        self.reference_mode = create_widget(
            annotation=str,
            label="Reference Mode",
            name="reference_mode",
            widget_type="ComboBox",
            options={"choices": ["Relative", "First Frame"]}
        )
        self.reference_mode.value = "Relative"

        self.output_path = create_widget(
            annotation=str,
            label="Output Directory",
            name="output_path",
            value=_DEFAULT_OUTPUT_PATH
        )

        self.multichannel_mode = create_widget(
            annotation=bool,
            label="Multichannel Registration Mode",
            name="multichannel_mode",
            value=False
        )

        self.correct_xy = create_widget(
            annotation=bool,
            label="XY Drift Correction",
            name="correct_xy",
            value=True
        )

        self.correct_z = create_widget(
            annotation=bool,
            label="Z Drift Correction",
            name="correct_z",
            value=True
        )

        self.correct_rotation = create_widget(
            annotation=bool,
            label="Rotation Correction",
            name="correct_rotation",
            value=True
        )

        self.crop_output = create_widget(
            annotation=bool,
            label="Crop Output",
            name="crop_output",
            value=False
        )

        self.export_data = create_widget(
            annotation=bool,
            label="Export CSV & Plots",
            name="export_data",
            value=True
        )

        # GPU acceleration toggle
        self.gpu_enabled = create_widget(
            annotation=bool,
            label="Enable GPU Acceleration",
            name="gpu_enabled",
            value=True
        )
        self.gpu_enabled.changed.connect(self._on_gpu_toggle)

        # GPU info display
        self.gpu_info_label = create_widget(
            annotation=str,
            label="Processing Backend",
            name="gpu_info",
            value=get_gpu_info(),
            options={"enabled": False}
        )

        # Progress bar and status
        self.status_label = create_widget(
            annotation=str,
            label="Status",
            name="status",
            value="Ready",
            options={"enabled": False}
        )

        self.progress_bar = ProgressBar(label="Progress")
        self.progress_bar.min = 0
        self.progress_bar.max = 100
        self.progress_bar.value = 0

        # Run button
        self.run_btn = PushButton(text="Run Registration")
        self.run_btn.changed.connect(self._on_run_clicked)

        # Add all widgets to container
        self.extend([
            self.image_layer,
            self.axes,
            self.ref_channel,
            self.normalize_channels,
            self.projection_type,
            self.reference_mode,
            self.output_path,
            self.multichannel_mode,
            self.correct_xy,
            self.correct_z,
            self.correct_rotation,
            self.crop_output,
            self.export_data,
            self.gpu_enabled,
            self.gpu_info_label,
            self.status_label,
            self.progress_bar,
            self.run_btn,
        ])

    def _on_gpu_toggle(self):
        """Handle GPU acceleration toggle."""
        enabled = self.gpu_enabled.value
        success = set_gpu_acceleration(enabled)
        # Update GPU info label with current backend
        self.gpu_info_label.value = get_gpu_info()
        if enabled and not success:
            self.status_label.value = "Warning: GPU not available, using CPU"
        else:
            backend = "GPU" if enabled else "CPU"
            self.status_label.value = f"Backend switched to {self.gpu_info_label.value}"

    def _on_cleanup_clicked(self):
        """Remove temporary zarr files from output directory."""
        output_dir = Path(self.output_path.value)
        
        if not output_dir.exists():
            return False
        
        tmp_files = ["tmp_data_1.zarr", "tmp_data_2.zarr"]
        removed_count = 0
        
        for tmp_file in tmp_files:
            tmp_path = output_dir / tmp_file
            if tmp_path.exists():
                try:
                    if tmp_path.is_dir():
                        shutil.rmtree(tmp_path)
                    else:
                        tmp_path.unlink()
                    removed_count += 1
                    print(f"Cleaned up: {tmp_path}")
                except Exception as e:
                    print(f"Error removing {tmp_path}: {e}")
        
        return removed_count > 0


    def _on_run_clicked(self):
        """Handle run button click."""
        if self.image_layer.value is None:
            self.status_label.value = "Error: No image layer selected"
            return

        # Disable button during processing
        self.run_btn.enabled = False
        self.status_label.value = "Starting registration..."
        self.progress_bar.value = 0

        # Reset timing variables
        self.start_time = time.time()
        self.last_update_time = self.start_time

        # Store output path for loading results later
        self.current_output_path = Path(self.output_path.value)
        self.current_layer_name = self.image_layer.value.name

        # Get image data
        image = self.image_layer.value.data

        # Start worker thread
        worker = self._run_registration(
            image=image,
            image_layer_name=self.current_layer_name,
            axes=self.axes.value,
            ref_channel=self.ref_channel.value,
            normalize_channels=self.normalize_channels.value,
            projection_type=self.projection_type.value,
            reference_mode=self.reference_mode.value,
            output_path=self.output_path.value,
            multichannel_mode=self.multichannel_mode.value,
            correct_xy=self.correct_xy.value,
            correct_z=self.correct_z.value,
            correct_rotation=self.correct_rotation.value,
            crop_output=self.crop_output.value,
            export_data=self.export_data.value,
        )

        worker.yielded.connect(self._on_progress)
        worker.returned.connect(self._on_complete)
        worker.errored.connect(self._on_error)
        worker.start()

    def _on_progress(self, progress_data):
        """Update progress bar and status with ETA."""
        step, total, message = progress_data
        progress_percent = int((step / total) * 100)
        self.progress_bar.value = progress_percent

        # Calculate ETA
        current_time = time.time()
        elapsed = current_time - self.start_time

        if step > 0 and step < total:
            avg_time_per_step = elapsed / step
            remaining_steps = total - step
            eta_seconds = avg_time_per_step * remaining_steps

            # Format ETA
            if eta_seconds < 60:
                eta_str = f"{eta_seconds:.0f}s"
            elif eta_seconds < 3600:
                eta_str = f"{eta_seconds/60:.1f}min"
            else:
                eta_str = f"{eta_seconds/3600:.1f}h"

            self.status_label.value = f"{message} [ETA: {eta_str}]"
        else:
            self.status_label.value = message

    def _on_complete(self, result):
        """Handle completion."""
        print(f"DEBUG: _on_complete called, self.viewer = {self.viewer}")
        print(f"DEBUG: self.viewer type = {type(self.viewer)}")
        self.progress_bar.value = 100

        # Calculate and display total time
        total_time = time.time() - self.start_time
        if total_time < 60:
            time_str = f"{total_time:.1f}s"
        elif total_time < 3600:
            time_str = f"{total_time/60:.1f}min"
        else:
            time_str = f"{total_time/3600:.2f}h"

        # Auto-cleanup temporary files
        cleanup_done = self._on_cleanup_clicked()
        cleanup_msg = " ✓ Temp files cleaned" if cleanup_done else ""

        self.status_label.value = f"✓ Registration complete! Total time: {time_str}{cleanup_msg}"
        self.run_btn.enabled = True

        # Load registered image from saved Zarr file instead of using returned array
        # This is more reliable for large images processed in worker thread
        if self.viewer is not None:
            try:
                zarr_path = self.current_output_path / f"{self.current_layer_name}_registered.zarr"
                if zarr_path.exists():
                    print(f"Loading registered image from: {zarr_path}")
                    # Load from zarr using dask for memory efficiency
                    registered_data = da.from_zarr(str(zarr_path))
                    # Compute to numpy for napari (napari handles dask arrays well too)
                    print(f"Adding registered image to napari viewer (shape: {registered_data.shape})")
                    self.viewer.add_image(registered_data, name=f"{self.current_layer_name}_registered")
                    print("✓ Registered image added to viewer")
                else:
                    print(f"Warning: Registered file not found at {zarr_path}")
            except Exception as e:
                print(f"Error loading registered image: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Warning: No viewer available to display result")

    def _on_error(self, error):
        """Handle error."""
        self.status_label.value = f"Error: {error}"
        self.run_btn.enabled = True
        self.progress_bar.value = 0
        print(f"Registration failed: {error}")
        import traceback
        traceback.print_exc()

    @thread_worker
    def _run_registration(
        self,
        image,
        image_layer_name,
        axes,
        ref_channel,
        normalize_channels,
        projection_type,
        reference_mode,
        output_path,
        multichannel_mode,
        correct_xy,
        correct_z,
        correct_rotation,
        crop_output,
        export_data,
    ):
        """Execute registration pipeline with fine-grained progress updates."""
        # Parse and validate axis string
        try:
            axes_string = str(axes).upper().strip()
        except Exception as e:
            raise ValueError(f"Invalid axis specification: {axes}. Error: {e}")

        # Calculate detailed steps with sub-steps
        # Base steps: data prep (2), temp storage cleanup (1), temp write (1)
        total_steps = 4
        if correct_xy:
            total_steps += 3  # detect, apply, write
        if correct_z:
            total_steps += 3  # detect, apply, write
        if crop_output:
            total_steps += 3  # crop, save, reload
        if correct_rotation:
            total_steps += 9  # detect alpha, apply alpha, write, detect beta, apply beta, write, detect gamma, apply gamma, write
        if export_data:
            total_steps += 2  # CSV export, plot generation
        total_steps += 2  # final export (compute + save)

        current_step = 0

        # Start timer
        start_time = time.time()

        # Convert to dask array
        yield (current_step, total_steps, "Step 1: Loading image data...")
        img = da.asarray(image)
        original_shape = img.shape  # Store original shape to restore at the end
        current_step += 1

        yield (current_step, total_steps, f"Step 2: Analyzing image shape {img.shape}...")
        current_step += 1

        # Convert to CTZYX format using flexible axis parser
        data, single_channel_mode, original_ndim = convert_to_ctzyx(img, axes_string)

        # Handle multichannel mode
        if multichannel_mode and not single_channel_mode:
            data = da.swapaxes(data, 0, 1)

        # Convert projection type and reference mode to lowercase for function calls
        projection_type_lower = projection_type.lower()
        reference_mode_lower = reference_mode.lower().replace(' ', '_')

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(output_dir)

        # Setup temp Zarr stores in output path
        tmp_path_1 = str(output_dir / "tmp_data_1.zarr")
        tmp_path_2 = str(output_dir / "tmp_data_2.zarr")
        tmp_path_read = tmp_path_1
        tmp_path_write = tmp_path_2

        # Clean up old temp directories
        yield (current_step, total_steps, "Cleaning up old temporary files...")
        for tmp_path_dir in [tmp_path_1, tmp_path_2]:
            if os.path.exists(tmp_path_dir):
                shutil.rmtree(tmp_path_dir)
        current_step += 1

        # Prepare data
        data = data.rechunk('auto')
        new_shape = data.chunksize

        yield (current_step, total_steps, f"Writing data to temporary storage ({data.nbytes / 1e9:.2f} GB)...")
        data = write_tmp_data_to_disk(tmp_path_write, data, new_shape)
        tmp_path_read, tmp_path_write = tmp_path_write, tmp_path_read
        current_step += 1

        # Initialize drift arrays
        xy_drift = np.array([[0, 0]])
        z_drift = np.array([[0]])
        alpha_xy = np.array([0])
        beta_zx = np.array([0])
        gamma_zy = np.array([0])

        tmp_data = data

        # XY correction
        if correct_xy:
            yield (current_step, total_steps, f"Detecting XY drift ({projection_type}, {reference_mode})...")
            xy_drift = get_xy_drift(tmp_data, ref_channel, projection_type_lower, reference_mode_lower, normalize_channels)
            current_step += 1

            yield (current_step, total_steps, "Applying XY drift correction...")
            tmp_data = apply_xy_drift(tmp_data, xy_drift)
            current_step += 1

            yield (current_step, total_steps, "Saving XY-corrected data...")
            tmp_data = write_tmp_data_to_disk(tmp_path_write, tmp_data, new_shape)
            tmp_path_read, tmp_path_write = tmp_path_write, tmp_path_read
            current_step += 1

        # Z correction
        if correct_z:
            yield (current_step, total_steps, f"Detecting Z drift ({projection_type}, {reference_mode})...")
            z_drift = get_z_drift(tmp_data, ref_channel, projection_type_lower, reference_mode_lower, normalize_channels)
            current_step += 1

            yield (current_step, total_steps, "Applying Z drift correction...")
            tmp_data = apply_z_drift(tmp_data, z_drift)
            current_step += 1

            yield (current_step, total_steps, "Saving Z-corrected data...")
            tmp_data = write_tmp_data_to_disk(tmp_path_write, tmp_data, new_shape)
            tmp_path_read, tmp_path_write = tmp_path_write, tmp_path_read
            current_step += 1

        # Cropping
        if crop_output:
            yield (current_step, total_steps, "Cropping image to valid region...")
            tmp_data = crop_data(tmp_data, xy_drift, z_drift)
            new_shape = (np.shape(tmp_data)[0], 1, np.shape(tmp_data)[-3],
                         np.shape(tmp_data)[-2], np.shape(tmp_data)[-1])
            current_step += 1

            yield (current_step, total_steps, "Saving cropped data...")
            # Use Zarr for cropped data as well
            crop_path = output_dir / "cropped_tmp_data.zarr"
            if crop_path.exists():
                shutil.rmtree(crop_path)
            tmp_data.rechunk(new_shape).to_zarr(str(crop_path))
            del tmp_data

            # Clean up old temp directory and replace with cropped version
            if os.path.exists(tmp_path_read):
                shutil.rmtree(tmp_path_read)
            shutil.move(str(crop_path), tmp_path_read)
            current_step += 1

            yield (current_step, total_steps, "Reloading cropped data...")
            tmp_data = read_tmp_data(tmp_path_read, new_shape)
            current_step += 1

        # Rotation correction - Sequential estimation and application
        if correct_rotation:
            # Alpha (XY plane) rotation
            yield (current_step, total_steps, f"Detecting XY plane rotation (alpha) ({projection_type}, {reference_mode})...")
            alpha_xy = get_rotation_alpha(tmp_data, ref_channel, projection_type_lower, reference_mode_lower, normalize_channels)
            current_step += 1

            yield (current_step, total_steps, "Applying XY plane rotation (alpha)...")
            tmp_data = apply_alpha_drift(tmp_data, alpha_xy)
            current_step += 1

            yield (current_step, total_steps, "Saving alpha-corrected data...")
            tmp_data = write_tmp_data_to_disk(tmp_path_write, tmp_data, new_shape)
            tmp_path_read, tmp_path_write = tmp_path_write, tmp_path_read
            current_step += 1

            # Beta (ZX plane) rotation
            yield (current_step, total_steps, f"Detecting ZX plane rotation (beta) ({projection_type}, {reference_mode})...")
            beta_zx = get_rotation_beta(tmp_data, ref_channel, projection_type_lower, reference_mode_lower, normalize_channels)
            current_step += 1

            yield (current_step, total_steps, "Applying ZX plane rotation (beta)...")
            tmp_data = apply_beta_drift(tmp_data, beta_zx)
            current_step += 1

            yield (current_step, total_steps, "Saving beta-corrected data...")
            tmp_data = write_tmp_data_to_disk(tmp_path_write, tmp_data, new_shape)
            tmp_path_read, tmp_path_write = tmp_path_write, tmp_path_read
            current_step += 1

            # Gamma (ZY plane) rotation
            yield (current_step, total_steps, f"Detecting ZY plane rotation (gamma) ({projection_type}, {reference_mode})...")
            gamma_zy = get_rotation_gamma(tmp_data, ref_channel, projection_type_lower, reference_mode_lower, normalize_channels)
            current_step += 1

            yield (current_step, total_steps, "Applying ZY plane rotation (gamma)...")
            tmp_data = apply_gamma_drift(tmp_data, gamma_zy)
            current_step += 1

            yield (current_step, total_steps, "Saving gamma-corrected data...")
            tmp_data = write_tmp_data_to_disk(tmp_path_write, tmp_data, new_shape)
            tmp_path_read, tmp_path_write = tmp_path_write, tmp_path_read
            current_step += 1

        # Export CSV and plots
        if export_data:
            yield (current_step, total_steps, "Exporting drift data to CSV...")

            # Export CSV
            x = pd.DataFrame({'x-drift': xy_drift[:, 0] if xy_drift.size > 0 else [0]})
            y = pd.DataFrame({'y-drift': xy_drift[:, 1] if xy_drift.size > 1 else [0]})
            z = pd.DataFrame({'z-drift': z_drift[:, 0] if z_drift.size > 0 else [0]})
            r_xy = pd.DataFrame({'rotation-xy': alpha_xy if alpha_xy.size > 0 else [0]})
            r_zx = pd.DataFrame({'rotation-zx': beta_zx if beta_zx.size > 0 else [0]})
            r_zy = pd.DataFrame({'rotation-zy': gamma_zy if gamma_zy.size > 0 else [0]})
            df = pd.concat([x, y, z, r_xy, r_zx, r_zy], axis=1)
            df = df.fillna(0)
            df.to_csv(str(output_dir / f"{image_layer_name}_drifts.csv"))
            current_step += 1

            # Generate plots
            yield (current_step, total_steps, "Generating drift analysis plots...")
            try:
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                fig.suptitle('Fast4DReg Drift Analysis', fontsize=16)

                # XY drift
                axes[0, 0].plot(df['x-drift'], label='X', marker='o')
                axes[0, 0].plot(df['y-drift'], label='Y', marker='s')
                axes[0, 0].set_xlabel('Frame')
                axes[0, 0].set_ylabel('Drift (pixels)')
                axes[0, 0].set_title('XY Drift')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)

                # Z drift
                axes[0, 1].plot(df['z-drift'], label='Z', marker='o', color='green')
                axes[0, 1].set_xlabel('Frame')
                axes[0, 1].set_ylabel('Drift (pixels)')
                axes[0, 1].set_title('Z Drift')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

                # Rotation XY
                axes[1, 0].plot(df['rotation-xy'], label='XY plane', marker='o', color='red')
                axes[1, 0].set_xlabel('Frame')
                axes[1, 0].set_ylabel('Rotation (degrees)')
                axes[1, 0].set_title('Rotation XY')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

                # Rotation ZX and ZY
                axes[1, 1].plot(df['rotation-zx'], label='ZX plane', marker='s', color='blue')
                axes[1, 1].plot(df['rotation-zy'], label='ZY plane', marker='^', color='purple')
                axes[1, 1].set_xlabel('Frame')
                axes[1, 1].set_ylabel('Rotation (degrees)')
                axes[1, 1].set_title('Rotation ZX/ZY')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(str(output_dir / f"{image_layer_name}_drift_analysis.png"), dpi=150, bbox_inches='tight')
                plt.close()
            except ImportError:
                print("matplotlib not installed, skipping plots")
            except Exception as e:
                print(f"Error generating plots: {e}")

            current_step += 1

        # Convert back to original axis order if needed (multichannel mode)
        if multichannel_mode and not single_channel_mode:
            tmp_data = da.swapaxes(tmp_data, 0, 1)

        # Export results - save to zarr without materializing in memory
        yield (current_step, total_steps, "Saving registered zarr...")
        zarr_path = output_dir / f"{image_layer_name}_registered.zarr"

        # Revert to original axis order
        registered_data = revert_to_original_axis_order(tmp_data, axes_string)

        # Determine optimal chunking for final output
        shape = registered_data.shape
        if len(shape) == 5:  # CTZYX or TCZYX
            chunks = (1, 1, shape[2], shape[3], shape[4])
        elif len(shape) == 4:
            chunks = (1, shape[1], shape[2], shape[3])
        else:
            chunks = None

        # Save directly to zarr from dask array (no compute, efficient streaming writes)
        if chunks:
            registered_data = registered_data.rechunk(chunks)
        registered_data.to_zarr(str(zarr_path), overwrite=True)
        current_step += 1

        elapsed = time.time() - start_time
        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        elif elapsed < 3600:
            time_str = f"{elapsed/60:.1f}min"
        else:
            time_str = f"{elapsed/3600:.2f}h"

        print("=" * 80)
        print("✓ REGISTRATION PIPELINE COMPLETE!")
        print(f"  Total processing time: {time_str}")
        print(f"  Input shape:  {original_shape}")
        print(f"  Output shape: {registered_data.shape}")
        print(f"  Saved to: {zarr_path}")
        print("=" * 80)

        yield (total_steps, total_steps, f"Registration complete! Total time: {time_str}")

        # Return metadata instead of large array to avoid thread communication issues
        return {
            'success': True,
            'output_path': str(zarr_path),
            'output_shape': registered_data.shape,
            'input_shape': original_shape,
            'elapsed_time': elapsed
        }


# Create widget instance function for napari
def Fast4DReg_widget(napari_viewer=None):
    """Factory function to create Fast4DReg widget."""
    print(f"DEBUG: Fast4DReg_widget factory called with napari_viewer = {napari_viewer}")
    print(f"DEBUG: napari_viewer type = {type(napari_viewer)}")

    # If viewer not passed, try to get current viewer
    if napari_viewer is None:
        import napari
        napari_viewer = napari.current_viewer()
        print(f"DEBUG: Retrieved current_viewer() = {napari_viewer}")

    return Fast4DRegWidget(napari_viewer)


