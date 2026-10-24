from utils import *
from sdf_project import *
import random
import open3d as o3d
import random
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from similaritymeasures import frechet_dist
from scipy.spatial import ConvexHull, KDTree
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def midpoint(p1, p2):
    return (p1 + p2) / 2

def get_parallelopiped_faces_from_vertices(vertices, flag=1):
    hull = ConvexHull(vertices)
    hull_points = vertices[hull.vertices]

    pca = PCA(n_components=3)
    pca.fit(hull_points)
    rotated = pca.transform(hull_points)

    min_bound = rotated.min(axis=0)
    max_bound = rotated.max(axis=0)

    corners = np.array([[x, y, z] for x in [min_bound[0], max_bound[0]]
                                  for y in [min_bound[1], max_bound[1]]
                                  for z in [min_bound[2], max_bound[2]]])

    box_corners = pca.inverse_transform(corners)

    kdtree = KDTree(vertices)
    _, indices = kdtree.query(box_corners)

    i = indices
    v = vertices

    faces_quad = [
        [i[0], i[1], i[3], i[2]],
        [i[4], i[5], i[7], i[6]],
        [i[0], i[1], i[5], i[4]],
        [i[2], i[3], i[7], i[6]],
        [i[0], i[2], i[6], i[4]],
        [i[1], i[3], i[7], i[5]],
    ]

    if flag == 1:
        return faces_quad

    elif flag == 2:
        new_faces = []
        tree = KDTree(vertices)

        for quad in faces_quad:
            a, b, c, d = [v[idx] for idx in quad]
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            cd = midpoint(c, d)
            da = midpoint(d, a)
            center = (a + b + c + d) / 4


            ab_i = tree.query(ab)[1]
            bc_i = tree.query(bc)[1]
            cd_i = tree.query(cd)[1]
            da_i = tree.query(da)[1]
            center_i = tree.query(center)[1]


            new_faces.append([quad[0], ab_i, center_i, da_i])
            new_faces.append([ab_i, quad[1], bc_i, center_i])


        return new_faces

import numpy as np
import open3d as o3d


try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def optimize_curve(V, F, N, D, high_curv_indices=[], vis=True, n_jobs=24):
    high_curv_indices = set(high_curv_indices)


    results = [_optimize_curve_one_cluster(i, v, f, n, d, high_curv_indices)
               for i, (v, f, n, d) in enumerate(zip(V, F, N, D))]


    lines_optimized = []
    for cluster_data in results:
        for pts, idx in cluster_data:
            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector(pts)
            line.lines = o3d.utility.Vector2iVector(idx)
            lines_optimized.append(line)

    if vis:
        o3d.visualization.draw_geometries(lines_optimized)
    return lines_optimized


def _optimize_curve_one_cluster(i, v, f, n, d, high_curv_indices):
    if len(high_curv_indices) > 0:
        try:
            if i in high_curv_indices:
                num_patches = 12
                patch_indices = get_parallelopiped_faces_from_vertices(v, flag=2)
            else:
                num_patches = 6
                patch_indices = get_parallelopiped_faces_from_vertices(v, flag=1)
        except Exception:
            num_patches = 1
            patch_indices = np.random.randint(0, len(v), size=(num_patches, 4))
    else:
        num_patches = 2
        patch_indices = compute_quad_indices(v)

    v = np.asarray(v)
    n_arr = np.asarray(n)
    d_arr = np.asarray(d)

    true_v = v - n_arr * d_arr
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(true_v)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(f)
    sdf_scene = get_sdf_scene(o3d_mesh)


    patch_indices = np.asarray(patch_indices)[:num_patches]

    idx_j = np.array([0, 1, 2, 3])
    idx_j1 = np.array([1, 2, 3, 0])


    all_i0 = patch_indices[:, idx_j]
    all_i1 = patch_indices[:, idx_j1]


    flat_i0 = all_i0.ravel()
    flat_i1 = all_i1.ravel()


    p1_adj = v[flat_i0] - n_arr[flat_i0] * d_arr[flat_i0]
    p2_adj = v[flat_i1] - n_arr[flat_i1] * d_arr[flat_i1]


    lengths = np.linalg.norm(p1_adj - p2_adj, axis=1)
    n_pts = np.maximum(5, (100 * lengths).astype(int))


    all_points, bounds = _sample_lines_vectorized(p1_adj, p2_adj, n_pts)


    projected = _project_sdf_batch(all_points, sdf_scene)


    return _build_line_data(projected, bounds)

def _build_line_data(projected, bounds):
    results = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if e - s < 2:
            continue
        seg_len = e - s
        points = np.ascontiguousarray(projected[s:e])
        lines = np.column_stack([np.arange(seg_len - 1), np.arange(1, seg_len)])
        results.append((points, lines))
    return results


def _sample_lines_vectorized(starts, ends, n_pts_array):
    n_pts_array = np.asarray(n_pts_array, dtype=np.int64)
    total = n_pts_array.sum()
    bounds = np.empty(len(n_pts_array) + 1, dtype=np.int64)
    bounds[0] = 0
    np.cumsum(n_pts_array, out=bounds[1:])


    seg_ids = np.repeat(np.arange(len(n_pts_array)), n_pts_array)

    local_idx = np.arange(total) - bounds[seg_ids]

    t = local_idx / np.maximum(n_pts_array[seg_ids] - 1, 1).astype(np.float64)

    points = starts[seg_ids] * (1 - t[:, None]) + ends[seg_ids] * t[:, None]
    return points.astype(np.float32), bounds


def _project_sdf_batch(points, sdf_scene, h=1e-4):
    points = np.asarray(points, dtype=np.float32)
    n = points.shape[0]


    sdf_center = sdf_scene.compute_signed_distance(points).numpy()


    offsets = np.zeros((6, n, 3), dtype=np.float32)
    for axis in range(3):
        offsets[2 * axis, :, axis] = h
        offsets[2 * axis + 1, :, axis] = -h

    all_offset_pts = (points[None, :, :] + offsets).reshape(-1, 3)
    all_sdf = sdf_scene.compute_signed_distance(all_offset_pts).numpy()
    all_sdf = all_sdf.reshape(6, n)


    grad = np.empty((n, 3), dtype=np.float32)
    for axis in range(3):
        grad[:, axis] = (all_sdf[2 * axis] - all_sdf[2 * axis + 1]) / (2 * h)

    if HAS_CUPY and n > 50000:

        sdf_g = cp.asarray(sdf_center)
        grad_g = cp.asarray(grad)
        pts_g = cp.asarray(points)
        norm_grad = cp.linalg.norm(grad_g, axis=1, keepdims=True) + 1e-8
        projected = pts_g - sdf_g[:, None] * (grad_g / norm_grad)
        return cp.asnumpy(projected)
    else:
        norm_grad = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-8
        projected = points - sdf_center[:, None] * (grad / norm_grad)
        return projected


def _build_linesets(projected, bounds):
    lines_out = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if e - s < 2:
            continue
        seg_len = e - s
        line_indices = np.column_stack([np.arange(seg_len - 1), np.arange(1, seg_len)])
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(projected[s:e])
        line.lines = o3d.utility.Vector2iVector(line_indices)
        lines_out.append(line)
    return lines_out


def compute_curve_curvature(curve_points):
    angles = []
    for i in range(1, len(curve_points) - 1):
        v1 = curve_points[i] - curve_points[i-1]
        v2 = curve_points[i+1] - curve_points[i]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            continue
        cos_angle = np.dot(v1, v2) / ((norm1 * norm2) + 1e-10)
        angle = np.arccos(cos_angle)
        angles.append(angle)
    return np.sum(angles)


def compute_frenet_frame(points):
    points = np.array(points)
    tangents, normals, binormals = [], [], []


    for i in range(1, len(points) - 1):
        t = points[i + 1] - points[i - 1]
        t_norm = np.linalg.norm(t) + 1e-8
        t = t / t_norm
        tangents.append(t)


        if abs(t[2]) < abs(t[0]) and abs(t[2]) < abs(t[1]):
            approx_normal = np.array([-t[1], t[0], 0])
        else:
            approx_normal = np.array([0, -t[2], t[1]])
        n = approx_normal / (np.linalg.norm(approx_normal) + 1e-8)
        normals.append(n)


        b = np.cross(t, n)
        binormals.append(b / (np.linalg.norm(b) + 1e-8))


    if len(tangents) > 0:
        tangents.insert(0, tangents[0])
        normals.insert(0, normals[0])
        binormals.insert(0, binormals[0])
        tangents.append(tangents[-1])
        normals.append(normals[-1])
        binormals.append(binormals[-1])
    else:

        tangents = [np.array([1, 0, 0])] * len(points)
        normals = [np.array([0, 1, 0])] * len(points)
        binormals = [np.array([0, 0, 1])] * len(points)

    return np.array(tangents), np.array(normals), np.array(binormals)

def get_length(stroke):
    pts = np.asarray(stroke)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())

def lineset_to_vertices(ls):
    return np.asarray(ls.points).tolist()

def vertices_to_lineset(vertices):
    pts = np.asarray(vertices)
    n   = len(pts)
    ls  = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)

    if n > 1:
        idx = np.arange(n - 1, dtype=np.int32)
        lines = np.stack([idx, idx + 1], axis=1)
        ls.lines = o3d.utility.Vector2iVector(lines)
    return ls

def lines_dist(chain1, chain2):
    mean1 = np.mean(chain1, axis=0)
    mean2 = np.mean(chain2, axis=0)
    return np.linalg.norm(mean1 - mean2)

def get_clustered_chains(new_chains, spatial_weight=0.7, angular_weight=0.3, distance_threshold=0.01):
    sim_array = np.zeros([len(new_chains), len(new_chains)])
    dist_array = np.zeros([len(new_chains), len(new_chains)])
    for i in range(len(new_chains)):
        for j in range(i + 1, len(new_chains)):
            v = np.array(new_chains[j][0]) - np.array(new_chains[i][0])
            x_trans = np.array(new_chains[i]) + v
            sim_array[i][j] = frechet_dist(x_trans, new_chains[j])


            dist_array[i][j] = lines_dist(new_chains[i], new_chains[j])
    sim_array = sim_array + sim_array.T
    dist_array = dist_array + dist_array.T
    return sim_array, dist_array

def get_final_chain(n_clusters, sim_array, dist_array, sample_num = 2):
    if n_clusters > len(sim_array):
        n_clusters = len(sim_array)
    clustering = AgglomerativeClustering(n_clusters=n_clusters,
                                         linkage='average').fit_predict(sim_array)
    final_chain = []
    for cluster_id in range(n_clusters):
        class_list = np.where(clustering == cluster_id)[0].tolist()
        if len(class_list) > 2:
            count = 0
            while count < sample_num and len(class_list) >= 2:
                subset = dist_array[class_list, :][:, class_list]
                max_index = np.unravel_index(subset.argmax(), subset.shape)
                sample = [class_list[max_index[0]], class_list[max_index[1]]]


                remove_items = []
                nearest = np.where(subset[max_index[0], :] < subset[max_index[0], :].mean())[0]
                remove_items.extend([class_list[item] for item in nearest])
                nearest = np.where(subset[max_index[1], :] < subset[max_index[1], :].mean())[0]
                remove_items.extend([class_list[item] for item in nearest])

                remove_items = set(remove_items)


                for item in remove_items:
                    class_list.remove(item)

                final_chain.extend(sample)
                count += 2
        else:
            final_chain.extend(class_list)
    return final_chain

def filter_similar_lines(chains, angular_thresh=0.1, spatial_thresh=0.05):
    if not chains:
        return []

    lengths = np.array([get_length(c) for c in chains])
    order   = np.argsort(-lengths)
    chains  = [chains[i] for i in order]


    dirs  = np.array([compute_direction_vector(c) for c in chains], dtype=np.float32)
    means = np.array([c.mean(axis=0)              for c in chains], dtype=np.float32)

    kept_indices = []

    for i in range(len(chains)):
        if not kept_indices:
            kept_indices.append(i)
            continue

        k = np.array(kept_indices)

        angular_sim = np.abs(dirs[k] @ dirs[i])
        angular_ok  = angular_sim > (1.0 - angular_thresh)

        if not angular_ok.any():
            kept_indices.append(i)
            continue


        candidate_k  = k[angular_ok]
        spatial_dist = np.linalg.norm(means[candidate_k] - means[i], axis=1)
        if (spatial_dist < spatial_thresh).any():
            continue

        kept_indices.append(i)

    return [chains[i] for i in kept_indices]

def compute_direction_vector(stroke):
    stroke = np.array(stroke)
    vec = stroke[-1] - stroke[0]
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.zeros_like(vec)
    return vec / norm


def post_process_curves(curve_line_sets, curvature_thresh=0.0, distance_threshold=0.05, vis=True, cluster_ratio=0.5):
    all_chains = [np.asarray(ls.points) for ls in curve_line_sets]

    filtered_chains = filter_similar_lines(all_chains, angular_thresh=0.2, spatial_thresh=0.05)
    curv = [vertices_to_lineset(stroke) for stroke in filtered_chains]
    if vis:
        o3d.visualization.draw_geometries(curv)
    return curv


def _humanify_compute_frames(points):
    n = points.shape[0]
    tangents = np.zeros_like(points)
    normals = np.zeros_like(points)
    binormals = np.zeros_like(points)
    tangents[:-1] = points[1:] - points[:-1]
    tangents[-1] = points[-1] - points[-2]
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8
    ref = np.array([0, 0, 1], dtype=float)
    if abs(np.dot(ref, tangents[0])) > 0.9:
        ref = np.array([0, 1, 0], dtype=float)
    normals[0] = ref - np.dot(ref, tangents[0]) * tangents[0]
    normals[0] /= np.linalg.norm(normals[0]) + 1e-8
    binormals[0] = np.cross(tangents[0], normals[0])
    for i in range(1, n):
        normals[i] = normals[i - 1] - np.dot(normals[i - 1], tangents[i]) * tangents[i]
        norm_len = np.linalg.norm(normals[i])
        if norm_len < 1e-8:
            normals[i] = np.cross(tangents[i], np.array([1, 0, 0]))
            if np.linalg.norm(normals[i]) < 1e-8:
                normals[i] = np.cross(tangents[i], np.array([0, 1, 0]))
            normals[i] /= np.linalg.norm(normals[i])
        else:
            normals[i] /= norm_len
        binormals[i] = np.cross(tangents[i], normals[i])
    return tangents, normals, binormals

def _humanify_one_segment(seg, seg_idx, a, b, noise_amplitude, helix_pitch):

    rng = np.random.default_rng(seg_idx)
    vertices = np.asarray(seg.points).copy()
    lines = np.asarray(seg.lines)


    allpt_idx = np.concatenate([lines[:, 0], lines[-1:, 1]])

    num_pts = allpt_idx.shape[0]

    if num_pts < 10:
        return seg


    _, normals, binormals = _humanify_compute_frames(vertices[allpt_idx])


    max_possible_segments = num_pts // a + 1
    jump_sizes = rng.integers(a, b, size=max_possible_segments)


    window_starts = np.cumsum(np.insert(jump_sizes[:-1] + 1, 0, 3))
    window_ends = window_starts + jump_sizes


    valid_mask = window_starts < num_pts - 3
    window_starts = window_starts[valid_mask]
    window_ends = np.minimum(window_ends[valid_mask], num_pts - 2)


    noise_factor_global = np.zeros(num_pts)
    phase_global = np.zeros(num_pts)

    for start, end in zip(window_starts, window_ends):
        length = end - start
        if length <= 0: continue


        mid = length // 2

        line = np.abs(np.linspace(-1, 1, length))
        noise_factor_global[start:end] = 1.0 - line


        phase_global[start:end] = np.linspace(0, helix_pitch * 2 * np.pi, length)


    offsets = (normals * np.cos(phase_global)[:, None] +
               binormals * np.sin(phase_global)[:, None])


    vertices[allpt_idx] += offsets * (noise_amplitude * noise_factor_global)[:, None]


    human_seg = o3d.geometry.LineSet()
    human_seg.points = o3d.utility.Vector3dVector(vertices)
    human_seg.lines = seg.lines
    return human_seg


def _humanify_one_segment_slow(seg, seg_idx, a, b, noise_amplitude, helix_pitch):
    random.seed(seg_idx)
    np.random.seed(seg_idx)
    vert_per_list = np.asarray(seg.points).copy()
    lineList = np.asarray(seg.lines)


    allpt_idx = np.concatenate([lineList[:, 0], lineList[-1:, 1]])

    mask = np.concatenate([[True], allpt_idx[1:] != allpt_idx[:-1]])
    allpt_idx = allpt_idx[mask]

    _, normals, binormals = _humanify_compute_frames(vert_per_list[allpt_idx])


    p_point = 3
    while p_point < allpt_idx.shape[0] - 3:
        nseg_perturb = random.randint(a, b)
        per_idx_1 = np.concatenate((np.array([p_point]), p_point + 1 + np.arange(nseg_perturb)))
        per_idx_1 = per_idx_1[per_idx_1 < allpt_idx.shape[0] - 2]
        if per_idx_1.shape[0] < 1:
            break
        per_idx_2 = allpt_idx[per_idx_1]
        mid_point = np.floor(len(per_idx_2) / 2)
        first_half = np.arange(1, mid_point + 1, 1)
        second_half = np.arange(mid_point, 0, -1)
        noise_factor = np.concatenate((first_half, second_half))
        if len(noise_factor) != len(per_idx_2):
            noise_factor = np.linspace(0, 1, len(per_idx_2))
        noise_factor = noise_factor / noise_factor.max()
        phase = np.linspace(0, helix_pitch * 2 * np.pi, len(per_idx_2))
        offsets = (normals[per_idx_2] * np.cos(phase)[:, None] +
                   binormals[per_idx_2] * np.sin(phase)[:, None]) * (noise_amplitude * noise_factor)[:, None]
        vert_per_list[per_idx_2, :] += offsets
        p_point = per_idx_1[-1] + 1

    human_seg = o3d.geometry.LineSet()
    human_seg.points = o3d.utility.Vector3dVector(vert_per_list)
    human_seg.lines = seg.lines
    return human_seg

def humanify(optimized_curves, a=10, b=30, noise_scale=0.1, noise_amplitude=0.01, helix_pitch=0.5,
             tube_radius=0.01, samples_per_circle=16, vis=True):
    human_curve = Parallel(n_jobs=-1, backend='threading')(
        delayed(_humanify_one_segment)(seg, seg_idx, a, b, noise_amplitude, helix_pitch)
        for seg_idx, seg in enumerate(optimized_curves)
    )
    if vis:
        o3d.visualization.draw_geometries(human_curve)
    return human_curve
