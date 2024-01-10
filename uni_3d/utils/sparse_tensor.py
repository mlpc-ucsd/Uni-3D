import torch


def to_dense(tensor, shape=None, min_coordinate=None, contract_stride=True, default_value=0.0):
    r"""Convert the :attr:`MinkowskiEngine.SparseTensor` to a torch dense
    tensor.
    Args:
        :attr:`shape` (torch.Size, optional): The size of the output tensor.
        :attr:`min_coordinate` (torch.IntTensor, optional): The min
        coordinates of the output sparse tensor. Must be divisible by the
        current :attr:`tensor_stride`. If 0 is given, it will use the origin for the min coordinate.
        :attr:`contract_stride` (bool, optional): The output coordinates
        will be divided by the tensor stride to make features spatially
        contiguous. True by default.
    Returns:
        :attr:`tensor` (torch.Tensor): the torch tensor with size `[Batch
        Dim, Feature Dim, Spatial Dim..., Spatial Dim]`. The coordinate of
        each feature can be accessed via `min_coordinate + tensor_stride *
        [the coordinate of the dense tensor]`.
        :attr:`min_coordinate` (torch.IntTensor): the D-dimensional vector
        defining the minimum coordinate of the output tensor.
        :attr:`tensor_stride` (torch.IntTensor): the D-dimensional vector
        defining the stride between tensor elements.
    """
    if min_coordinate is not None:
        assert isinstance(min_coordinate, torch.IntTensor)
        assert min_coordinate.numel() == tensor._D
    if shape is not None:
        assert isinstance(shape, torch.Size)
        assert len(shape) == tensor._D + 2  # batch and channel
        if shape[1] != tensor._F.size(1):
            shape = torch.Size([shape[0], tensor._F.size(1), *[s for s in shape[2:]]])

    # Exception handling for empty tensor
    if tensor.__len__() == 0:
        assert shape is not None, "shape is required to densify an empty tensor"
        return (
            torch.zeros(shape, dtype=tensor.dtype, device=tensor.device),
            torch.zeros(tensor._D, dtype=torch.int32, device=tensor.device),
            tensor.tensor_stride,
        )

    # Use int tensor for all operations
    tensor_stride = torch.IntTensor(tensor.tensor_stride).to(tensor.device)

    # New coordinates
    batch_indices = tensor.C[:, 0]

    if min_coordinate is None:
        min_coordinate, _ = tensor.C.min(0, keepdim=True)
        min_coordinate = min_coordinate[:, 1:]
        if not torch.all(min_coordinate >= 0):
            raise ValueError(
                f"Coordinate has a negative value: {min_coordinate}. Please provide min_coordinate argument"
            )
        coords = tensor.C[:, 1:]
    elif isinstance(min_coordinate, int) and min_coordinate == 0:
        coords = tensor.C[:, 1:]
    else:
        min_coordinate = min_coordinate.to(tensor.device)
        if min_coordinate.ndim == 1:
            min_coordinate = min_coordinate.unsqueeze(0)
        coords = tensor.C[:, 1:] - min_coordinate

    assert (
        min_coordinate % tensor_stride
    ).sum() == 0, "The minimum coordinates must be divisible by the tensor stride."

    if coords.ndim == 1:
        coords = coords.unsqueeze(1)

    # return the contracted tensor
    if contract_stride:
        coords = torch.div(coords, tensor_stride, rounding_mode="floor")

    nchannels = tensor.F.size(1)
    if shape is None:
        size = coords.max(0)[0] + 1
        shape = torch.Size(
            [batch_indices.max() + 1, nchannels, *size.cpu().numpy()]
        )

    dense_F = torch.full(shape, dtype=tensor.F.dtype, device=tensor.F.device, fill_value=default_value)

    tcoords = coords.t().long()
    batch_indices = batch_indices.long()
    exec(
        "dense_F[batch_indices, :, "
        + ", ".join([f"tcoords[{i}]" for i in range(len(tcoords))])
        + "] = tensor.F"
    )

    tensor_stride = torch.IntTensor(tensor.tensor_stride)
    return dense_F, min_coordinate, tensor_stride
