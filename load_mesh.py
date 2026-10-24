import open3d as o3d
import numpy as np

def load_mesh_and_fit_unit(path, vis=True):
    mesh = o3d.io.read_triangle_mesh(path)
    if vis:
        o3d.visualization.draw_geometries([mesh])
    min_bound = mesh.get_min_bound()
    max_bound = mesh.get_max_bound()

    center = (min_bound + max_bound) / 2
    scale = np.max(max_bound - min_bound)

    mesh_translated = mesh.translate(-center)
    mesh_scaled = mesh_translated.scale(1 / scale, center=np.array([0, 0, 0]))
    mesh = mesh_scaled
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    if vis:
        o3d.visualization.draw_geometries([mesh])
    return mesh
