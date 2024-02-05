import torch
import torch.nn.functional as F

def _thicken_grid(grid, grid_dims, frustum_mask):
    device = frustum_mask.device
    offsets = torch.nonzero(torch.ones(3, 3, 3, device=device)).long()
    locs_grid = grid.nonzero(as_tuple=False)
    locs = locs_grid.unsqueeze(1).repeat(1, 27, 1)
    locs += offsets
    locs = locs.view(-1, 3)
    masks = ((locs >= 0) & (locs < torch.as_tensor(grid_dims, device=device))).all(-1)
    locs = locs[masks]

    thicken = torch.zeros(grid_dims, dtype=torch.bool, device=device)
    thicken[locs[:, 0], locs[:, 1], locs[:, 2]] = True
    # frustum culling
    thicken = thicken & frustum_mask

    return thicken


def prepare_instance_masks_thicken(instances, semantic_mapping, distance_field, frustum_mask, 
                                   iso_value=1.0, truncation=3.0, downsample_factor: int=1):
    assert isinstance(downsample_factor, int) and 256 % downsample_factor == 0
    grid_dims = [256, 256, 256]
    need_rescale = downsample_factor != 1
    if need_rescale:
        grid_dims = (torch.as_tensor(grid_dims) // downsample_factor).tolist()
        frustum_mask = F.interpolate(frustum_mask[None, None].float(), 
                                     size=grid_dims, mode="nearest").squeeze(0, 1).bool()

    instance_information = {}

    for instance_id, semantic_class in semantic_mapping.items():
        instance_mask: torch.Tensor = (instances == instance_id)
        instance_distance_field = torch.full_like(instance_mask, dtype=torch.float, fill_value=truncation)
        instance_distance_field[instance_mask] = distance_field.squeeze()[instance_mask]
        instance_distance_field_masked = instance_distance_field.abs() < iso_value

        if need_rescale:
            instance_distance_field_masked = F.max_pool3d(instance_distance_field_masked[None, None].float(), 
                                                          kernel_size=downsample_factor+1, stride=downsample_factor, padding=1).squeeze(0, 1).bool()

        # instance_grid = instance_grid & frustum_mask
        instance_grid = _thicken_grid(instance_distance_field_masked, grid_dims, frustum_mask)
        instance_grid: torch.Tensor = instance_grid.to(torch.device("cpu"), non_blocking=True)
        instance_information[instance_id] = instance_grid, semantic_class

    return instance_information
