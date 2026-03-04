"""Axis order utilities shared by API and widget."""

import dask.array as da


def parse_axis_order(axis_string):
    """Parse axis order string and convert to CTZYX format.

    Handles variable-length axis orders like:
    - "CTZYX" or "TZCYX" (5D)
    - "TZYX", "ZCYX", "CZYX" (4D)
    - "ZYX", "TYX", "CYX" (3D)
    - "YX" (2D, will add C, T, Z)

    Returns:
        tuple: (normalized_order_string, has_channel, has_time, has_z)
    """
    axis_string = axis_string.upper().strip()

    # Validate characters
    valid_chars = set("CTZYX")
    if not all(c in valid_chars for c in axis_string):
        raise ValueError(
            f"Invalid axis order: {axis_string}. Use only C, T, Z, Y, X"
        )

    # Check for duplicates
    if len(axis_string) != len(set(axis_string)):
        raise ValueError(f"Duplicate axes in: {axis_string}")

    has_c = "C" in axis_string
    has_t = "T" in axis_string
    has_z = "Z" in axis_string
    has_y = "Y" in axis_string
    has_x = "X" in axis_string

    # Y and X should always be present
    if not (has_y and has_x):
        raise ValueError(f"Axis order must contain Y and X: {axis_string}")

    return axis_string, has_c, has_t, has_z


def convert_to_ctzyx(img, axis_string):
    """Convert image data to CTZYX format based on axis string.

    Adds singleton dimensions for missing axes (C, T, Z) and reorders
    to canonical CTZYX format.

    Args:
        img: Input dask or numpy array
        axis_string: Axis order string (e.g., "TZYX", "ZYX", "CTZYX")

    Returns:
        Tuple of (converted_image, single_channel_mode, original_length)
    """
    axis_string = axis_string.upper().strip()
    original_length = len(axis_string)

    # Parse axis string
    axis_order, has_c, has_t, has_z = parse_axis_order(axis_string)

    single_channel_mode = False
    # Convert to dask array early for reliable operations
    data = da.asarray(img)

    # Handle different input shapes and convert to CTZYX
    if len(data.shape) != original_length:
        raise ValueError(
            f"Shape mismatch: Image has {len(data.shape)} dimensions but "
            f"axis order '{axis_order}' specifies {original_length} dimensions"
        )

    # Find positions of each axis in input
    axis_pos = {axis: i for i, axis in enumerate(axis_order)}

    # Add singleton dimensions for missing axes using dask
    # Insert in reverse order of CTZYX to maintain indices
    if not has_c:
        data = da.expand_dims(data, axis=0)  # Add C at position 0
        # Increment all axes that come after position 0
        for ax in axis_pos:
            axis_pos[ax] += 1
        has_c = True
        axis_pos["C"] = 0

    if not has_t:
        # Insert T at position 1 (after C)
        if has_c:
            data = da.expand_dims(data, axis=1)
            # Increment all axes at position 1 or greater
            for ax in axis_pos:
                if axis_pos[ax] >= 1:
                    axis_pos[ax] += 1
        else:
            data = da.expand_dims(data, axis=0)
            # Increment all axes at position 0 or greater
            for ax in axis_pos:
                axis_pos[ax] += 1
        has_t = True
        axis_pos["T"] = 1

    if not has_z:
        # Insert Z at position 2 (after C, T)
        data = da.expand_dims(data, axis=2)
        # Increment all axes at position 2 or greater
        for ax in axis_pos:
            if axis_pos[ax] >= 2:
                axis_pos[ax] += 1
        has_z = True
        axis_pos["Z"] = 2

    # Now reorder to CTZYX using dask
    target_order = "CTZYX"
    axes_permutation = []
    for target_axis in target_order:
        if target_axis in axis_pos:
            axes_permutation.append(axis_pos[target_axis])

    if len(axes_permutation) == len(target_order):
        data = da.moveaxis(data, axes_permutation, range(len(target_order)))

    # Check if single channel mode
    if data.shape[0] == 1:
        single_channel_mode = True

    return data, single_channel_mode, original_length


def revert_to_original_axis_order(data, original_axis_string):
    """Convert data from CTZYX back to original axis order.

    Removes added singleton dimensions and reorders axes back to user's original
    specification.

    Args:
        data: Data in CTZYX format (5D dask array)
        original_axis_string: Original axis order provided by user (e.g., "TZYX",
            "ZYX")

    Returns:
        Data reordered to match original axis string and with singleton dims
        removed
    """
    original_axis_string = original_axis_string.upper().strip()

    # Current positions of each axis in CTZYX (5D)
    ctzyx_pos = {"C": 0, "T": 1, "Z": 2, "Y": 3, "X": 4}
    ctzyx_order = "CTZYX"

    # Parse original axis order
    axis_order, has_c, has_t, has_z = parse_axis_order(original_axis_string)

    # Find which dimensions were added (singleton dims that weren't in original)
    dims_to_remove = []
    for ax in ctzyx_order:
        if ax not in original_axis_string:
            dims_to_remove.append(ctzyx_pos[ax])

    # Remove added singleton dimensions (in reverse order to preserve indices)
    result = data
    for ax in sorted(dims_to_remove, reverse=True):
        if result.shape[ax] == 1:  # Only squeeze if it's a singleton
            result = da.squeeze(result, axis=ax)

    # After removing singleton dims, axes may need reordering
    remaining_axes = [ax for ax in ctzyx_order if ax in original_axis_string]

    if len(remaining_axes) > 1 and remaining_axes != list(original_axis_string):
        # Move each axis from its current position to the position required by
        # original_axis_string so output order matches the user input.
        src_pos = [remaining_axes.index(ax) for ax in original_axis_string]
        dst_pos = list(range(len(original_axis_string)))
        result = da.moveaxis(result, src_pos, dst_pos)

    return result
