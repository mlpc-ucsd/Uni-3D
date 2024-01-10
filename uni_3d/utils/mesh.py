from typing import Union, List, Tuple, Optional
import os
import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import KDTree
import marching_cubes as mc


def create_color_palette():
    return [
        (0, 0, 0),
        (174, 199, 232),		# wall
        (152, 223, 138),		# floor
        (31, 119, 180), 		# cabinet
        (255, 187, 120),		# bed
        (188, 189, 34), 		# chair
        (140, 86, 75),  		# sofa
        (255, 152, 150),		# table
        (214, 39, 40),  		# door
        (197, 176, 213),		# window
        (148, 103, 189),		# bookshelf
        (196, 156, 148),		# picture
        (23, 190, 207), 		# counter
        (178, 76, 76),
        (247, 182, 210),		# desk
        (66, 188, 102),
        (219, 219, 141),		# curtain
        (140, 57, 197),
        (202, 185, 52),
        (51, 176, 203),
        (200, 54, 131),
        (92, 193, 61),
        (78, 71, 183),
        (172, 114, 82),
        (255, 127, 14), 		# refrigerator
        (91, 163, 138),
        (153, 98, 156),
        (140, 153, 101),
        (158, 218, 229),		# shower curtain
        (100, 125, 154),
        (178, 127, 135),
        (120, 185, 128),
        (146, 111, 194),
        (44, 160, 44),  		# toilet
        (112, 128, 144),		# sink
        (96, 207, 209),
        (227, 119, 194),		# bathtub
        (213, 92, 176),
        (94, 106, 211),
        (82, 84, 163),  		# otherfurn
        (100, 85, 144),
        (172, 172, 172),
    ]


def lookup_colors(labels: np.array, color_palette: List = None) -> np.array:
    if color_palette is None:
        color_palette = np.array(create_color_palette())

    color_volume = color_palette[labels]
    return color_volume


def coords_multiplication(matrix, points):
    """
    matrix: 4x4
    points: nx3
    """
    if isinstance(matrix, torch.Tensor):
        points = torch.cat([points.t(), torch.ones((1, points.shape[0]), device=matrix.device)])
        return torch.mm(matrix, points).t()[:, :3]
    elif isinstance(matrix, np.ndarray):
        points = np.concatenate([np.transpose(points), np.ones((1, points.shape[0]))])
        return np.transpose(np.dot(matrix, points))[:, :3]


def write_ply(vertices: Union[np.array, torch.Tensor], colors: Union[np.array, torch.Tensor, List, Tuple],
              faces: Union[np.array, torch.Tensor], output_file: os.PathLike) -> None:
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()

    if isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()

    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()

    if colors is not None:
        if isinstance(colors, list) or isinstance(colors, tuple):
            colors = np.ones_like(vertices) * np.array(colors)

    if faces is None:
        faces = []

    with open(output_file, "w") as file:
        file.write("ply \n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(vertices):d}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")

        if colors is not None:
            file.write("property uchar red\n")
            file.write("property uchar green\n")
            file.write("property uchar blue\n")

        if faces is not None:
            file.write(f"element face {len(faces):d}\n")
            file.write("property list uchar uint vertex_indices\n")
        file.write("end_header\n")

        if colors is not None:
            for vertex, color in zip(vertices, colors):
                file.write(f"{vertex[0]:f} {vertex[1]:f} {vertex[2]:f} ")
                file.write(f"{int(color[0]):d} {int(color[1]):d} {int(color[2]):d}\n")
        else:
            for vertex in vertices:
                file.write(f"{vertex[0]:f} {vertex[1]:f} {vertex[2]:f}\n")

        for face in faces:
            file.write(f"3 {face[0]:d} {face[1]:d} {face[2]:d}\n")


def write_distance_field(distance_field: Union[np.array, torch.Tensor], labels: Optional[Union[np.array, torch.Tensor]],
                         output_file: os.PathLike, iso_value: float = 1.0, truncation: float = 3.0,
                         color_palette=None, transform=None) -> None:
    if isinstance(distance_field, torch.Tensor):
        distance_field = distance_field.detach().cpu().numpy()

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    vertices, triangles = get_mesh(distance_field, iso_value, truncation)
    if labels is not None:
        labels_kd = KDTree(np.stack(labels.nonzero(), axis=-1))
        labels = labels.astype(np.uint32)
        color_volume = lookup_colors(labels, color_palette)
        neighbor_inds = labels_kd.query(vertices)[1]
        neighbors = labels_kd.data[neighbor_inds].astype(int)
        colors = color_volume[neighbors[:, 0], neighbors[:, 1], neighbors[:, 2]]
    else:
        colors = None

    if transform is not None:
        if isinstance(transform, torch.Tensor):
            transform = transform.detach().cpu().numpy()

        vertices = coords_multiplication(transform, vertices)

    write_ply(vertices, colors, triangles, output_file)


def write_pointcloud(points: Union[np.array, torch.Tensor], colors: Union[np.array, torch.Tensor, List, Tuple],
                     output_file: os.PathLike) -> None:
    write_ply(points, colors, None, output_file)


def write_semantic_pointcloud(points: Union[np.array, torch.Tensor], labels: Union[np.array, torch.Tensor],
                              output_file: os.PathLike, color_palette=None) -> None:
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    colors = lookup_colors(labels, color_palette)
    write_pointcloud(points, colors, output_file)


def get_mesh(distance_field: np.array, iso_value: float = 1.0, truncation: float = 3.0) -> Tuple[np.array, np.array]:
    vertices, triangles = mc.marching_cubes(distance_field, iso_value, truncation)
    return vertices, triangles


def read_ply(ply_file):
    with open(ply_file, "rb") as file:
        ply_data = PlyData.read(file)

    points = []
    colors = []
    indices = []

    for x, y, z, r, g, b in ply_data["vertex"]:
        points.append([x, y, z])
        colors.append([r, g, b])

    for face in ply_data["face"]:
        indices.append([face[0][0], face[0][1], face[0][2]])

    points = np.array(points)
    colors = np.array(colors)
    indices = np.array(indices)

    return points, indices, colors

