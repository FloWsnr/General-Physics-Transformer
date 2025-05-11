def _calculate_strides(
    dimension_size: int,
    num_layers: int,
    default_stride: int = 2,
) -> list[int]:
    """
    Calculate the strides for each layer to achieve the desired downsampling.

    Parameters
    ----------
    dimension_size : int
        The size of the dimension to downsample (time, height, or width)
    num_layers : int
        The number of layers in the encoder
    default_stride : int
        The default stride to use if the dimension size is not divisible by the total downsampling factor

    Returns
    -------
    list[int]
        A list of stride values for each layer
    """
    # Initialize all strides to 2 (default downsampling factor)
    strides = [default_stride] * num_layers

    # Calculate the total downsampling factor with all strides=2
    total_downsampling = default_stride**num_layers

    # If we need more downsampling, increase some strides
    if total_downsampling < dimension_size:
        remaining_factor = dimension_size // total_downsampling

        # Find the number of additional powers of 2 needed
        additional_power = 0
        while 2**additional_power < remaining_factor:
            additional_power += 1

        # Distribute the additional downsampling across layers
        for i in range(additional_power):
            if i < num_layers:
                strides[i] *= 2

    # If we need less downsampling, decrease some strides
    elif total_downsampling > dimension_size:
        # Find the number of powers of 2 we need to remove
        excess_power = 0
        while total_downsampling // (2**excess_power) > dimension_size:
            excess_power += 1

        # Adjust strides to achieve the desired downsampling
        for i in range(excess_power):
            if i < num_layers:
                strides[num_layers - i - 1] = 1

    return strides
