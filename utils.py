import time
import numpy as np
import scipy
from sklearn.cluster import KMeans, AgglomerativeClustering
import open3d as o3d
import scipy.sparse
import trimesh
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.spatial import ConvexHull

def tic():
    return time.perf_counter()

def tock(t0, label):
    print(f"{label}: {time.perf_counter() - t0:.3f} s")


def check_mesh_connectivity_cluster(vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    adjacency_edges = mesh.face_adjacency
    num_faces = len(mesh.triangles)

    adjacency_matrix = scipy.sparse.lil_matrix((num_faces, num_faces))
    for edge in adjacency_edges:
        i, j = edge
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1

    n_components, labels = connected_components(adjacency_matrix, directed=False)

    return n_components, labels

def subdivide_cluster_if_disconnected(V_cluster, F_cluster):
    n_components, comp_labels = check_mesh_connectivity_cluster(V_cluster, F_cluster)
    if n_components == 1:
        return [(V_cluster, F_cluster)]

    subclusters = []
    for comp in range(n_components):
        face_indices = [i for i, label in enumerate(comp_labels) if label == comp]
        sub_F = [F_cluster[i] for i in face_indices]
        vertex_set = set()
        for face in sub_F:
            vertex_set.update(face)
        vertex_list = sorted(list(vertex_set))
        mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(vertex_list)}
        sub_V = [V_cluster[i] for i in vertex_list]
        sub_F_remap = [[mapping[v] for v in face] for face in sub_F]
        subclusters.append((np.array(sub_V), np.array(sub_F_remap)))
    return subclusters

def bezier_curve(control_points, t):
    p1, p2, p3, p4 = control_points
    return (1 - t)**3 * p1 + 3 * (1 - t)**2 * t * p2 + 3 * (1 - t) * t**2 * p3 + t**3 * p4

def create_linesets(line_sets, points):
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(points)
    lines.lines = o3d.utility.Vector2iVector([[0,1], [1,2], [2,3], [3,0]])
    line_sets.append(lines)
    return line_sets


def compute_normal(v1, v2, v3):
    edge1 = v2 - v1
    edge2 = v3 - v1
    normal = np.cross(edge1, edge2)
    return normal / np.linalg.norm(normal)

def compute_centroid(v1, v2, v3):
    return (v1 + v2 + v3) / 3


def compute_geodesic_distance(mesh):
    adjacency_edges = mesh.face_adjacency
    num_faces = len(mesh.faces)


    adjacency_matrix = scipy.sparse.lil_matrix((num_faces, num_faces))
    for edge in adjacency_edges:
        i, j = edge
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1


    return scipy.sparse.csgraph.shortest_path(adjacency_matrix, directed=False)

def compute_angle(mesh, face_idx):
    face_vertices = mesh.vertices[mesh.faces[face_idx]]
    angles = []
    for i in range(3):
        v1, v2, v3 = face_vertices[i], face_vertices[(i+1)%3], face_vertices[(i+2)%3]
        cosine_angle = np.dot(v2-v1, v3-v1) / (np.linalg.norm(v2-v1) * np.linalg.norm(v3-v1))
        angles.append(np.arccos(cosine_angle, -1.0, 1.0))


    defect = 2 * np.pi - sum(angles)
    return defect

def compute_euler_characteristic(V, F):
    E = set()
    for f in F:
        E.add(tuple(sorted([f[0], f[1]])))
        E.add(tuple(sorted([f[1], f[2]])))
        E.add(tuple(sorted([f[2], f[0]])))
    return len(V) - len(E) + len(F)

def compute_best_fit_plane(points):
    mean = np.mean(points, axis=0)
    centered = points - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)

    basis = Vt[:2].T
    return mean, basis

def project_points(points, mean, basis):
    return np.dot(points - mean, basis)

def minimum_bounding_rectangle(points):
    hull = ConvexHull(points, qhull_options='QJ')
    hull_points = points[hull.vertices]

    best_area = np.inf
    best_rect = None
    for i in range(len(hull_points)):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % len(hull_points)]
        edge = p2 - p1
        angle = np.arctan2(edge[1], edge[0])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])

        rot_points = np.dot(hull_points, R)
        min_x, max_x = np.min(rot_points[:, 0]), np.max(rot_points[:, 0])
        min_y, max_y = np.min(rot_points[:, 1]), np.max(rot_points[:, 1])
        area = (max_x - min_x) * (max_y - min_y)

        if area < best_area:
            best_area = area
            rect = np.array([
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y]
            ])
            best_rect = np.dot(rect, R.T)
    return best_rect

def select_mesh_vertices_from_rect(rect_2d, hull_points, hull_indices):
    selected = []
    for corner in rect_2d:
        dists = np.linalg.norm(hull_points - corner, axis=1)
        sorted_idx = np.argsort(dists)
        for idx in sorted_idx:
            if hull_indices[idx] not in selected:
                selected.append(hull_indices[idx])
                break
    if len(list(set(selected))) < 4:
        for idx in hull_indices:
            selected.append(idx)
            if len(selected) == 4:
                break
    return selected

def order_vertices_cyclic(points_2d, indices):
    selected_points = points_2d[indices]
    centroid = np.mean(selected_points, axis=0)
    angles = np.arctan2(selected_points[:, 1] - centroid[1], selected_points[:, 0] - centroid[0])
    order = np.argsort(angles)
    return [indices[i] for i in order]

def compute_oriented_quadrilateral_mesh_vertices(points):
    mean, basis = compute_best_fit_plane(points)
    points_2d = project_points(points, mean, basis)

    hull = ConvexHull(points_2d, qhull_options='QJ')
    hull_indices = hull.vertices
    hull_points = points_2d[hull_indices]

    rect_2d = minimum_bounding_rectangle(points_2d)

    selected_indices = select_mesh_vertices_from_rect(rect_2d, hull_points, hull_indices)

    ordered_indices = order_vertices_cyclic(points_2d, selected_indices)

    return ordered_indices
def select_edge_midpoints(points, quad_indices):
    mid_indices = []
    for i in range(len(quad_indices)):
        j = (i + 1) % len(quad_indices)
        midpoint = (points[quad_indices[i]] + points[quad_indices[j]]) / 2.0

        dists = np.linalg.norm(points - midpoint, axis=1)

        dists[quad_indices[i]] = np.inf
        dists[quad_indices[j]] = np.inf

        closest_index = np.argmin(dists)
        mid_indices.append(closest_index)
    return mid_indices

def compute_quad_indices(points):
    corners = compute_oriented_quadrilateral_mesh_vertices(points)
    A, B, C, D = corners

    mid_indices = select_edge_midpoints(points, corners)
    E = mid_indices[0]
    G = mid_indices[2]

    quad1 = [A, E, G, D]
    quad2 = [B, E, G, C]

    indices_array = np.array([quad1, quad2])
    return indices_array
