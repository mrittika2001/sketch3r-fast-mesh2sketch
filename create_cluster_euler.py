import numpy as np
import math
from skimage.filters import threshold_otsu
import open3d as o3d
import torch
import trimesh
import scipy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

def compute_geodesic_features(mesh, n_landmarks=128):
    num_faces = len(mesh.faces)
    centroids = mesh.triangles_center
    adj = mesh.face_adjacency
    weights = np.linalg.norm(centroids[adj[:, 0]] - centroids[adj[:, 1]], axis=1)

    graph = scipy.sparse.coo_matrix(
        (np.concatenate([weights, weights]),
         (np.concatenate([adj[:, 0], adj[:, 1]]),
          np.concatenate([adj[:, 1], adj[:, 0]]))),
        shape=(num_faces, num_faces)
    ).tocsr()


    n_landmarks = min(n_landmarks, num_faces)
    landmark_ids = np.random.choice(num_faces, n_landmarks, replace=False)
    dist_landmarks = scipy.sparse.csgraph.shortest_path(
        graph, directed=False, indices=landmark_ids
    )

    return dist_landmarks.T.astype(np.float32)

def to_torch(arr, dtype=dtype, device=device):
    if torch.is_tensor(arr):
        return arr.to(device=device, dtype=dtype)
    return torch.tensor(np.asarray(arr), dtype=dtype, device=device)

def to_numpy(t):
    if torch.is_tensor(t):
        return t.detach().cpu().numpy()
    return np.asarray(t)

def compute_face_normals(V_np, F_np):
    V = to_torch(V_np)
    F = to_torch(F_np, dtype=torch.long)
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = torch.cross(v1 - v0, v2 - v0, dim=1)
    n = n / torch.norm(n, dim=1, keepdim=True).clamp(min=1e-9)
    return n


def compute_face_centroids(V_np, F_np):
    V = to_torch(V_np)
    F = to_torch(F_np, dtype=torch.long)
    return (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0


def kmeans(X_torch, n_clusters=100, iters=100, verbose=False):
    N, D = X_torch.shape
    rnd_idx = torch.randperm(N, device=device)[:n_clusters]
    centroids = X_torch[rnd_idx].clone()

    for it in range(iters):

        x2    = (X_torch ** 2).sum(dim=1, keepdim=True)
        c2    = (centroids ** 2).sum(dim=1, keepdim=True).t()
        dist2 = x2 + c2 - 2.0 * (X_torch @ centroids.t())
        labels = torch.argmin(dist2, dim=1)


        new_centroids = torch.zeros(n_clusters, D, dtype=X_torch.dtype, device=device)
        counts        = torch.zeros(n_clusters,    dtype=X_torch.dtype, device=device)
        new_centroids.scatter_add_(0, labels.unsqueeze(1).expand(N, D), X_torch)
        counts.scatter_add_(0, labels, torch.ones(N, dtype=X_torch.dtype, device=device))


        empty = counts == 0
        if empty.any():
            n_empty = int(empty.sum().item())
            rand_pts = X_torch[torch.randint(0, N, (n_empty,), device=device)]
            new_centroids[empty] = rand_pts
            counts[empty] = 1.0

        new_centroids /= counts.unsqueeze(1)

        shift = torch.norm(new_centroids - centroids, dim=1).mean().item()
        centroids = new_centroids
        if verbose and it % 10 == 0:
            print(f"[KMeans] it {it}, shift {shift:.6f}")
        if shift < 1e-6:
            break
    return labels.cpu().numpy(), centroids


def build_face_adjacency(F_np):
    F = np.asarray(F_np, dtype=np.int64)
    edges = {}
    for fi, f in enumerate(F):
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            edges.setdefault(key, []).append(fi)
    Fcount = len(F)
    adj = [set() for _ in range(Fcount)]
    unique_edges = []
    for (u, v), faces in edges.items():
        unique_edges.append((u, v))
        if len(faces) > 1:
            for i in range(len(faces)):
                for j in range(i + 1, len(faces)):
                    adj[faces[i]].add(faces[j])
                    adj[faces[j]].add(faces[i])
    return adj, np.array(unique_edges, dtype=np.int64)


def check_mesh_connectivity_cluster(V_sub, F_sub):
    F = np.asarray(F_sub, dtype=np.int64)
    Fcount = len(F)


    v0 = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    v1 = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    fi = np.tile(np.arange(Fcount, dtype=np.int64), 3)

    lo = np.minimum(v0, v1)
    hi = np.maximum(v0, v1)


    max_v = int(hi.max()) + 1
    key = lo * max_v + hi


    order = np.argsort(key, kind='stable')
    key_sorted = key[order]
    fi_sorted  = fi[order]


    boundaries = np.flatnonzero(np.diff(key_sorted)) + 1
    groups = np.split(fi_sorted, boundaries)


    adj = [[] for _ in range(Fcount)]
    for g in groups:
        if len(g) == 2:

            adj[g[0]].append(g[1])
            adj[g[1]].append(g[0])
        elif len(g) > 2:

            for ii in range(len(g)):
                for jj in range(ii + 1, len(g)):
                    adj[g[ii]].append(g[jj])
                    adj[g[jj]].append(g[ii])


    comp_labels = -np.ones(Fcount, dtype=np.int32)
    comp_id = 0
    for i in range(Fcount):
        if comp_labels[i] != -1:
            continue
        q = [i]
        comp_labels[i] = comp_id
        qi = 0
        while qi < len(q):
            u = q[qi]; qi += 1
            for v in adj[u]:
                if comp_labels[v] == -1:
                    comp_labels[v] = comp_id
                    q.append(v)
        comp_id += 1
    return comp_id, comp_labels


def subdivide_cluster_if_disconnected(V_sub, F_sub):
    F = np.asarray(F_sub, dtype=np.int64)
    unique_vertices, inverse = np.unique(F.flatten(), return_inverse=True)
    Vlocal = np.asarray(V_sub)[unique_vertices]
    Flocal = inverse.reshape(-1, 3)
    ncomp, comp_labels = check_mesh_connectivity_cluster(Vlocal, Flocal)
    if ncomp <= 1:
        return [(Vlocal, Flocal)]
    parts = []
    for c in range(ncomp):
        idxs = np.where(comp_labels == c)[0]
        faces_c = Flocal[idxs]
        uverts, inv = np.unique(faces_c.flatten(), return_inverse=True)
        parts.append((Vlocal[uverts], inv.reshape(-1, 3)))
    return parts


def compute_euler_characteristic(V_sub, F_sub):
    F = np.asarray(F_sub, dtype=np.int64)
    Vn = np.asarray(V_sub).shape[0]
    e01 = np.sort(F[:, [0, 1]], axis=1)
    e12 = np.sort(F[:, [1, 2]], axis=1)
    e20 = np.sort(F[:, [2, 0]], axis=1)
    all_edges = np.concatenate([e01, e12, e20], axis=0)
    max_v = int(all_edges.max()) + 1
    packed = all_edges[:, 0].astype(np.int64) * max_v + all_edges[:, 1].astype(np.int64)
    En = np.unique(packed).shape[0]
    return int(Vn - En + len(F))


def estimate_curvature_with_euler(vertices, faces):
    nv = len(vertices)
    V  = torch.tensor(vertices, dtype=torch.float64, device=device)
    F  = torch.tensor(faces,    dtype=torch.long,    device=device)

    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cross       = torch.linalg.cross(v1 - v0, v2 - v0)
    face_area   = torch.linalg.norm(cross, dim=1) * 0.5


    area = torch.zeros(nv, dtype=torch.float64, device=device)
    fa3  = (face_area / 3.0).repeat(3)
    area.scatter_add_(0, F.reshape(-1), fa3)


    ii, jj, kk = F[:, 0], F[:, 1], F[:, 2]
    def _cot(a, b, c):
        ea, eb = b - a, c - a
        dot  = (ea * eb).sum(dim=1)
        crs  = torch.linalg.norm(torch.linalg.cross(ea, eb), dim=1).clamp(min=1e-12)
        return dot / crs

    cot_i = _cot(V[ii], V[jj], V[kk])
    cot_j = _cot(V[jj], V[kk], V[ii])
    cot_k = _cot(V[kk], V[ii], V[jj])


    rows = torch.cat([ii, jj,  jj, kk,  kk, ii,   ii, jj, jj, kk, kk, ii])
    cols = torch.cat([jj, ii,  kk, jj,  ii, kk,   ii, jj, jj, kk, kk, ii])
    nf   = F.shape[0]

    vals = torch.cat([
        -cot_k, -cot_k,
        -cot_i, -cot_i,
        -cot_j, -cot_j,
         cot_k,  cot_k,
         cot_i,  cot_i,
         cot_j,  cot_j,
    ])


    idx = torch.stack([rows, cols], dim=0)
    L   = torch.sparse_coo_tensor(idx, vals, (nv, nv), dtype=torch.float64, device=device)
    Lz  = torch.mv(L, V[:, 2])

    curvature = torch.zeros(nv, dtype=torch.float64, device=device)
    mask = area > 1e-8
    curvature[mask] = torch.abs(Lz[mask] / area[mask])

    chi        = compute_euler_characteristic(vertices, faces)
    total_area = area.sum().item() or 1.0
    phi        = ((curvature * area).sum().item() / total_area) + 2.0 * math.pi * chi
    return float(phi)


def _process_one_raw_cluster(faces_list, V_np):
    faces = np.array(faces_list, dtype=np.int64)
    unique_vertices, inverse = np.unique(faces.flatten(), return_inverse=True)
    sub_V = V_np[unique_vertices]
    sub_F = inverse.reshape(-1, 3)
    ncomp, _ = check_mesh_connectivity_cluster(sub_V, sub_F)
    if ncomp > 1:
        parts = subdivide_cluster_if_disconnected(sub_V, sub_F)
        return [(Vp, Fp, compute_euler_characteristic(Vp, Fp)) for Vp, Fp in parts]
    return [(sub_V, sub_F, compute_euler_characteristic(sub_V, sub_F))]


def merge_clusters_to_limit(clustered_V, clustered_F, clustered_euler_char,
                                max_clusters=300, min_vertices=3):
    num_c = len(clustered_V)


    if num_c <= max_clusters:
        sizes = np.array([len(v) for v in clustered_V])
        if (sizes >= min_vertices).all():
            return clustered_V, clustered_F, clustered_euler_char


    if num_c < 10000:
        return _merge_sequential(clustered_V, clustered_F, clustered_euler_char,
                                max_clusters, min_vertices)
    else:
        print(f"[Hierarchical merge] {num_c:,} clusters → {max_clusters}")
        return _merge_hierarchical(clustered_V, clustered_F, clustered_euler_char,
                                   max_clusters, min_vertices)


def _merge_sequential(clustered_V, clustered_F, clustered_euler_char,
                      max_clusters, min_vertices):
    num_c = len(clustered_V)
    sizes = np.array([len(v) for v in clustered_V])
    centroids = np.array([v.mean(axis=0) for v in clustered_V])

    active_indices = np.arange(num_c)
    is_active = np.ones(num_c, dtype=bool)
    parent = np.arange(num_c)

    def get_merge_candidates():
        small_mask = (sizes < min_vertices) & is_active
        if small_mask.any():
            return np.where(small_mask)[0][0]
        if is_active.sum() > max_clusters:
            return active_indices[is_active][np.argmin(sizes[is_active])]
        return None

    while True:
        idx_s = get_merge_candidates()
        if idx_s is None:
            break

        c_s = centroids[idx_s]
        search_mask = is_active.copy()
        search_mask[idx_s] = False

        active_lookup = active_indices[search_mask]
        diffs = centroids[active_lookup] - c_s
        dists_sq = np.einsum('ij,ij->i', diffs, diffs)
        idx_n = active_lookup[np.argmin(dists_sq)]

        new_size = sizes[idx_n] + sizes[idx_s]
        centroids[idx_n] = (centroids[idx_n] * sizes[idx_n] + centroids[idx_s] * sizes[idx_s]) / new_size
        sizes[idx_n] = new_size
        parent[idx_s] = idx_n
        is_active[idx_s] = False


    final_clusters = {}
    for i in range(num_c):
        root = i
        while parent[root] != root:
            root = parent[root]
        final_clusters.setdefault(root, []).append(i)

    V_out, F_out, E_out = [], [], []
    for root, children in final_clusters.items():
        V_group = [clustered_V[c] for c in children]
        F_group = []
        offset = 0
        for c in children:
            F_group.append(clustered_F[c] + offset)
            offset += len(clustered_V[c])

        V_final = np.vstack(V_group)
        F_final = np.vstack(F_group)
        V_out.append(V_final)
        F_out.append(F_final)
        E_out.append(compute_euler_characteristic(V_final, F_final))

    return V_out, F_out, E_out


def _merge_hierarchical(clustered_V, clustered_F, clustered_euler_char,
                        max_clusters, min_vertices):
    import time
    t_start = time.time()

    current_V = clustered_V
    current_F = clustered_F
    current_E = clustered_euler_char

    iteration = 0
    prev_len = len(current_V)
    while len(current_V) > max_clusters:
        iteration += 1
        n_current = len(current_V)


        centroids = np.array([V.mean(axis=0) for V in current_V])


        if n_current > max_clusters * 10:
            target_this_iter = max(max_clusters, n_current // 10)
        else:
            target_this_iter = max_clusters

        print(f"  Iteration {iteration}: {n_current:,} → {target_this_iter:,} clusters", end=" ")


        grid_resolution = int(np.cbrt(target_this_iter)) + 1


        bbox_min = centroids.min(axis=0)
        bbox_max = centroids.max(axis=0)
        bbox_range = bbox_max - bbox_min + 1e-10


        normalized = (centroids - bbox_min) / bbox_range
        grid_idx = (normalized * grid_resolution).astype(np.int32)
        grid_idx = np.clip(grid_idx, 0, grid_resolution - 1)


        grid_hash = (grid_idx[:, 0] * grid_resolution * grid_resolution +
                     grid_idx[:, 1] * grid_resolution +
                     grid_idx[:, 2])


        unique_cells = np.unique(grid_hash)
        merged_V, merged_F, merged_E = [], [], []

        for cell_id in unique_cells:
            cell_mask = grid_hash == cell_id
            cell_indices = np.where(cell_mask)[0]

            if len(cell_indices) == 1:

                idx = cell_indices[0]
                merged_V.append(current_V[idx])
                merged_F.append(current_F[idx])
                merged_E.append(current_E[idx])
            else:

                V_list = [current_V[i] for i in cell_indices]
                F_list = [current_F[i] for i in cell_indices]


                V_merged = np.vstack(V_list)


                F_merged_list = []
                offset = 0
                for V, F in zip(V_list, F_list):
                    F_merged_list.append(F + offset)
                    offset += len(V)
                F_merged = np.vstack(F_merged_list)

                merged_V.append(V_merged)
                merged_F.append(F_merged)
                merged_E.append(compute_euler_characteristic(V_merged, F_merged))

        current_V = merged_V
        current_F = merged_F
        current_E = merged_E

        print(f"→ {len(current_V):,} clusters ({time.time()-t_start:.1f}s elapsed)")
        if prev_len == len(current_V):
            print(f"Hierarchical merge stuck! Breaking while loop. {time.time()-t_start:.1f}s total")
            return current_V, current_F, current_E
        else:
            prev_len = len(current_V)

    print(f"  Hierarchical merge complete: {time.time()-t_start:.1f}s total")

    return current_V, current_F, current_E


def mesh_clustering_with_euler(V, F, N_clusters=500, max_clusters=300):
    V_np = np.asarray(V, dtype=np.float64)
    F_np = np.asarray(F, dtype=np.int64)
    Fcount = F_np.shape[0]

    face_normals_t = compute_face_normals(V_np, F_np)


    mesh = trimesh.Trimesh(vertices=V_np, faces=F_np, process=False)

    geo_feat = compute_geodesic_features(mesh, n_landmarks=128)
    geo_feat = torch.from_numpy(geo_feat).to(device)
    geo_feat = (geo_feat - geo_feat.min()) / (geo_feat.max() - geo_feat.min() + 1e-12)

    normal_dists_feat = face_normals_t

    alpha, beta = 0.3, 1.0
    feat = torch.cat([alpha * geo_feat, beta * normal_dists_feat], dim=1)
    n_clusters = max(1, min(N_clusters, Fcount))
    labels_np, _ = kmeans(feat, n_clusters=n_clusters, iters=100)


    labels_t = torch.tensor(labels_np, dtype=torch.long, device=device)
    F_t = torch.tensor(F_np, dtype=torch.long, device=device)
    order_t = torch.argsort(labels_t, stable=True)
    labels_sorted = labels_t[order_t]
    F_sorted = F_t[order_t]
    boundaries_t = torch.cat([
        torch.tensor([0], device=device),
        (torch.where(labels_sorted[1:] != labels_sorted[:-1])[0] + 1),
        torch.tensor([Fcount], device=device),
    ])
    boundaries_np = boundaries_t.cpu().numpy()
    F_sorted_np = F_sorted.cpu().numpy()

    raw_clusters = [F_sorted_np[boundaries_np[k]:boundaries_np[k+1]]
                    for k in range(len(boundaries_np) - 1)]
    nested = [_process_one_raw_cluster(faces_list, V_np) for faces_list in raw_clusters]

    clustered_V, clustered_F, clustered_euler = [], [], []
    for parts in nested:
        for sub_V, sub_F, euler in parts:
            clustered_V.append(sub_V)
            clustered_F.append(sub_F)
            clustered_euler.append(euler)

    return merge_clusters_to_limit(clustered_V, clustered_F, clustered_euler, max_clusters=max_clusters)


def return_cluster(vertices, faces, N_cluster, vis=True):
    clustered_V, clustered_F, clustered_euler = mesh_clustering_with_euler(
        vertices, faces, N_clusters=N_cluster
    )

    V_list, F_list, N_list, D_list, phi_list = [], [], [], [], []

    for i, (v, f, _) in enumerate(zip(clustered_V, clustered_F, clustered_euler)):
        np.random.seed(i)
        v = np.asarray(v, dtype=np.float64)
        f = np.asarray(f, dtype=np.int64)
        nv = len(v)


        V_t = torch.tensor(v, dtype=torch.float32, device=device)
        F_t = torch.tensor(f, dtype=torch.long, device=device)
        v0t, v1t, v2t = V_t[F_t[:, 0]], V_t[F_t[:, 1]], V_t[F_t[:, 2]]
        fn_t = torch.linalg.cross(v1t - v0t, v2t - v0t)
        fn_t = fn_t / (torch.linalg.norm(fn_t, dim=1, keepdim=True).clamp(min=1e-12))

        vn_t = torch.zeros(nv, 3, dtype=torch.float32, device=device)
        for ci in range(3):
            vn_t.scatter_add_(0, F_t[:, ci].unsqueeze(1).expand(-1, 3), fn_t)
        vn_t = vn_t / (torch.linalg.norm(vn_t, dim=1, keepdim=True).clamp(min=1e-12))
        vn = vn_t.cpu().numpy().astype(np.float64)


        phi_val = estimate_curvature_with_euler(v, f)


        delta = np.random.uniform(0.0, 0.1, size=v.shape)

        V_list.append(v + delta * vn)
        F_list.append(f)
        N_list.append(vn)
        D_list.append(delta)
        phi_list.append(np.nan_to_num(phi_val))

    phi = np.array(phi_list, dtype=np.float64)

    rng = phi.max() - phi.min()
    phi = (phi - phi.min()) / rng if rng > 1e-12 else np.zeros_like(phi)

    thresh = threshold_otsu(phi) if len(phi) > 0 else 1.0
    high_curv_indices = np.where(phi >= thresh)[0]
    low_curv_indices  = np.where(phi <  thresh)[0]

    if vis:
        meshes = []
        for v, f, n, _ in zip(V_list, F_list, N_list, D_list):
            m = o3d.geometry.TriangleMesh()
            m.vertices       = o3d.utility.Vector3dVector(v)
            m.triangles      = o3d.utility.Vector3iVector(f)
            m.vertex_normals = o3d.utility.Vector3dVector(n)
            meshes.append(m)
        o3d.visualization.draw_geometries(meshes)

    return [V_list, F_list, N_list, D_list, high_curv_indices, low_curv_indices]
