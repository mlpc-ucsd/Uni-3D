from datetime import datetime
from typing import List, Any

import numpy as np

import torch
from torch.nn import functional as F


class Compose:
    def __init__(self, transforms: List[Any], profiling: bool = False) -> None:
        self.transforms = transforms
        self.profiling = profiling

    def __call__(self, data, *args, **kwargs):
        if self.profiling:
            data, timings = self.call_with_profiling(data, *args, **kwargs)
            return data, timings
        else:
            for transform in self.transforms:
                data = transform(data, *args, **kwargs)

            return data

    def call_with_profiling(self, data, *args, **kwargs):
        timings = {}

        total_start = datetime.now()

        for transform in self.transforms:
            start = datetime.now()
            data = transform(data, *args, **kwargs)
            end = datetime.now()

            name = type(transform).__name__
            timings[name] = (end - start).total_seconds()

        total_end = datetime.now()
        name = type(self).__name__
        timings[name] = (total_end - total_start).total_seconds()

        return data, timings


class FromNumpy:
    def __call__(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        tensor = torch.from_numpy(data)
        return tensor


class ToTensor:
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.dtype is None:
            tensor = torch.as_tensor(data)
        else:
            tensor = torch.as_tensor(data, dtype=self.dtype)
        return tensor


class ToOccupancy:
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def __call__(self, distance_field: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        one = torch.ones([1], device=distance_field.device)
        zero = torch.zeros([1], device=distance_field.device)
        occupancy_grid = torch.where(torch.abs(distance_field) < self.threshold, one, zero)
        return occupancy_grid


class ToTDF:
    def __init__(self, truncation):
        self.truncation = truncation

    def __call__(self, distance_field: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        distance_field = torch.abs(distance_field)
        distance_field = torch.clip(distance_field, 0, self.truncation)
        return distance_field


class ToBinaryMask:
    def __init__(self, threshold: float, compare_function=torch.lt):
        self.threshold = threshold
        self.compare_function = compare_function

    def __call__(self, mask: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        mask = self.compare_function(mask, self.threshold)
        return mask


class Absolute:
    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return torch.abs(x)


class NormalizeOccupancyGrid:
    def __call__(self, occupancy_grid: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        normalized = (occupancy_grid * 2) - 1
        return normalized


def compute_dimension_difference(dimensions: torch.Size, volume: torch.Tensor) -> torch.Tensor:
    source_dimensions = volume.dim()
    if source_dimensions == 5:  # B,C,XYZ
        difference = torch.tensor(dimensions) - torch.tensor(volume.shape[2:])
    elif source_dimensions == 4:  # B,XYZ
        difference = torch.tensor(dimensions) - torch.tensor(volume.shape[1:])
    elif source_dimensions == 3:  # XYZ
        difference = torch.tensor(dimensions) - torch.tensor(volume.shape)
    else:
        difference = torch.zeros(len(dimensions))
    return difference.float()


class ResizeTrilinear:
    def __init__(self, factor: float, mode: str = "trilinear"):
        self.factor = factor
        self.mode = mode
        self.mode_args = {
            "trilinear": {
                "recompute_scale_factor": False,
                "align_corners": True
            },
            "nearest": {
                "recompute_scale_factor": True
            }
        }

    def __call__(self, volume: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        old_dim = volume.dim()
        while volume.dim() < 5:
            volume = volume.unsqueeze(0)

        mode_args = self.mode_args.get(self.mode, {})
        resized = F.interpolate(volume, scale_factor=self.factor, mode=self.mode, **mode_args)

        while resized.dim() > old_dim:
            resized.squeeze_(0)

        return resized


class ResizeMax:
    def __init__(self, kernel_size, stride, padding):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, volume: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        old_dtype = volume.type()
        old_dim = volume.dim()

        while volume.dim() < 5:
            volume = volume.unsqueeze(0)

        volume = volume.type(torch.float)

        resized = F.max_pool3d(volume, self.kernel_size, self.stride, self.padding)
        resized = resized.type(old_dtype)

        while resized.dim() > old_dim:
            resized.squeeze_(0)

        return resized
