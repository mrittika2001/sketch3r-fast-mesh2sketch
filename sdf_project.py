import numpy as np
import trimesh
import open3d as o3d
import numpy as np

def sample_line(start, end, num_points=100):
    return np.linspace(start, end, num_points)


def get_sdf_scene(mesh_legacy):
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)
    return scene


def compute_signed_distance_and_gradient(p, scene, h=1e-4, m=True):
    if not hasattr(scene, "compute_signed_distance"):
        scene = get_sdf_scene(scene)
    if m:
        sdf = scene.compute_signed_distance(np.array([p], dtype=np.float32)).numpy()[0]
        grad = np.zeros(3)
        for i in range(3):
            dp = np.zeros(3)
            dp[i] = h
            f_plus = scene.compute_signed_distance(np.array([p + dp], dtype=np.float32)).numpy()[0]
            f_minus = scene.compute_signed_distance(np.array([p - dp], dtype=np.float32)).numpy()[0]
            grad[i] = (f_plus - f_minus) / (2 * h)
        return sdf, grad
    else:
        sdf = scene.compute_signed_distance(np.array([p], dtype=np.float32)).numpy()[0]
        return sdf


def compute_signed_distance_and_gradient_batched(points, scene, h=1e-4):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, 3)
    n = points.shape[0]
    sdf = scene.compute_signed_distance(points).numpy()
    grad = np.zeros((n, 3), dtype=np.float32)
    for i in range(3):
        dp = np.zeros((n, 3), dtype=np.float32)
        dp[:, i] = h
        sdf_plus = scene.compute_signed_distance(points + dp).numpy()
        sdf_minus = scene.compute_signed_distance(points - dp).numpy()
        grad[:, i] = (sdf_plus - sdf_minus) / (2 * h)
    return sdf, grad


def sample_line_and_project_sdf(start, end, num_points, mesh, tol=1e-8, curve_control=[], sdf_scene=None):
    line_points = sample_line(start, end, num_points)
    if sdf_scene is None:
        sdf_scene = get_sdf_scene(mesh)
    sdf, grad = compute_signed_distance_and_gradient_batched(line_points, sdf_scene)
    norm_grad = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-8
    projected_points = line_points - sdf[:, None] * (grad / norm_grad)
    return [projected_points]


def sample_lines_and_project_sdf_batch(segments, sdf_scene, mesh=None):
    if not segments:
        return []
    if sdf_scene is None:
        sdf_scene = get_sdf_scene(mesh)
    all_points = []
    bounds = [0]
    for (start, end, num_points) in segments:
        pts = sample_line(start, end, num_points)
        all_points.append(pts)
        bounds.append(bounds[-1] + len(pts))
    points = np.concatenate(all_points, axis=0).astype(np.float32)
    sdf, grad = compute_signed_distance_and_gradient_batched(points, sdf_scene)
    norm_grad = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-8
    projected = points - sdf[:, None] * (grad / norm_grad)
    return [projected[bounds[i]:bounds[i + 1]] for i in range(len(segments))]
