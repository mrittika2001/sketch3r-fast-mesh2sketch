import gc
import numpy as np
from tqdm import tqdm
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed

try:
    import torch
    _TORCH_AVAILABLE = True
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    _TORCH_AVAILABLE = False
    _DEVICE = None


def segment_mesh_by_angle(mesh, angle_threshold=np.radians(3), tolerance=np.radians(2), n_jobs=-1, use_gpu=True):
    mesh.compute_triangle_normals()
    normals = np.asarray(mesh.triangle_normals)
    norm_magnitudes = np.linalg.norm(normals, axis=1)
    norm_magnitudes[norm_magnitudes == 0] = 1
    normalized_normals = normals / norm_magnitudes[:, np.newaxis]

    clusters, _, _ = mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    sorted_indices = np.argsort(clusters)
    change_points = np.flatnonzero(np.diff(clusters[sorted_indices])) + 1
    list_of_clusters = np.split(sorted_indices, change_points)
    num_clusters = len(list_of_clusters)


    sorted_normals = normalized_normals[sorted_indices]


    shift_change_points = np.insert(change_points, 0, 0)


    summed_cluster_normals = np.add.reduceat(sorted_normals, shift_change_points, axis=0)


    cluster_norms = np.linalg.norm(summed_cluster_normals, axis=1, keepdims=True)

    cluster_normals = np.divide(summed_cluster_normals, cluster_norms,
                                out=np.zeros_like(summed_cluster_normals),
                                where=cluster_norms != 0)
    ram_usage = cluster_normals.shape[0]**2 * 4 / 10**9
    print(f"Projected adjacency size in memory: {ram_usage} GB")
    if use_gpu and _TORCH_AVAILABLE and _DEVICE.type == "cuda" and num_clusters > 0 and ram_usage < 1:

        cn = torch.from_numpy(cluster_normals).float().to(_DEVICE)
        dots = torch.mm(cn, cn.t()).clamp(-1.0, 1.0)
        angles = torch.acos(dots)
        torch.cuda.empty_cache()
        adjacency = (torch.abs(angles - angle_threshold) <= tolerance).cpu().numpy()
        del angles
        gc.collect()
        torch.cuda.empty_cache()
    else:
        dots = np.dot(cluster_normals, cluster_normals.T).clip(-1.0, 1.0)
        angles = np.arccos(dots)
        adjacency = np.abs(angles - angle_threshold) <= tolerance


    graph = csr_matrix(adjacency)
    num_components, labels = connected_components(graph, directed=False)


    face_component_labels = labels[clusters]


    face_indices = np.arange(len(face_component_labels))
    sorted_face_indices = face_indices[np.argsort(face_component_labels)]


    change_points = np.flatnonzero(np.diff(face_component_labels[sorted_face_indices])) + 1
    disjoint_sets = np.split(sorted_face_indices, change_points)


    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    segments = []


    lookup = np.empty(len(vertices), dtype=np.int32)

    for face_indices in disjoint_sets:
        if len(face_indices) == 0:
            continue


        sub_triangles = triangles[face_indices]


        unique_v_idx = np.unique(sub_triangles)


        lookup[unique_v_idx] = np.arange(len(unique_v_idx))


        new_faces = lookup[sub_triangles]


        new_vertices = vertices[unique_v_idx]

        segments.append((new_vertices, new_faces))


    return segments, len(segments)
