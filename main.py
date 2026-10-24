import os
import h5py
import time
import json
import shutup
shutup.please()
import argparse
import traceback
import numpy as np
import pyvista as pv
import open3d as o3d
from utils import tic, tock
from load_mesh import load_mesh_and_fit_unit
from create_cluster_euler import return_cluster
from create_graph_cluster import segment_mesh_by_angle
from optimize_bezier import optimize_curve, post_process_curves, humanify
from meshlib_decim import decimate


def process_file(path, args):
    print(f"Processing {path}")
    total_start = time.perf_counter()
    t = tic()
    mesh_pr = load_mesh_and_fit_unit(path, vis=False)
    tock(t, f"Load + Normalize")

    t = tic()
    target_faces = 3000
    mesh = decimate(mesh_pr, target_faces, max_error=0.01)
    num_verts, num_faces = np.array(mesh_pr.vertices).shape[0], np.array(mesh_pr.triangles).shape[0]
    num_verts2, num_faces2 = np.array(mesh.vertices).shape[0], np.array(mesh.triangles).shape[0]
    tock(t, f"Decimation {num_verts}->{num_verts2} ({num_verts2/num_verts:.3f}) {num_faces}->{num_faces2} ({num_faces2/num_faces:.3f})")

    t = tic()
    _, num_clusters = segment_mesh_by_angle(mesh)
    tock(t, "Segmentation")

    t = tic()
    V, F, N, D, high_curv_indices, _ = return_cluster(
        np.array(mesh.vertices),
        np.array(mesh.triangles),
        max(args.min_clusters, num_clusters),
        vis=False
    )
    tock(t, "Cluster Extraction")

    t = tic()
    lines_optimized = optimize_curve(V, F, N, D, high_curv_indices=high_curv_indices, vis=False)
    tock(t, "Curve Optimization")

    t = tic()
    optimized_lines = post_process_curves(lines_optimized, vis=False)
    sparse_network = optimized_lines

    tock(t, "Post-processing")

    t = tic()
    human_lines = humanify(optimized_lines, noise_scale=0.0, vis=False)
    tock(t, "Humanification")
    print(f"Total processing time: {time.perf_counter() - total_start:.3f} s")
    print("---------------------------------------------------------------------------------------------------------------")
    return mesh_pr, mesh, lines_optimized, sparse_network, human_lines

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Abstraction of 3D Objects")
    parser.add_argument("--mesh", required=True, help="Path of Mesh to Process")
    parser.add_argument("--out_root", required=True, help="Where to save?")
    parser.add_argument("--min_clusters", default=10, help="Minimum Number of cluster for mesh")
    parser.add_argument("--save", action='store_true', default=False, help="It saves the generated curve network")
    parser.add_argument("--vis", action='store_true', default=False, help="It saves the generated curve network")
    args = parser.parse_args()
    path = args.mesh


    save_path = ''.join(map(str, path.split('.')[:-1])) + '_all_output.h5'
    save_path = os.path.join(str(args.out_root), '/'.join(save_path.split("/")[-2:]))
    os.makedirs('/'.join(save_path.split("/")[:-1]), exist_ok=True)


    if os.path.exists(save_path):
        print(f"File {save_path} already exists, skipping processing.")
        exit(0)
    try:
        mesh_pr, mesh, lines_optimized, sparse_lines, human_lines = process_file(path, args)
        all_lines_optimized = o3d.geometry.LineSet()
        for i in lines_optimized:
            all_lines_optimized += i

        all_sparse_lines = o3d.geometry.LineSet()
        for i in sparse_lines:
            all_sparse_lines += i

        all_human_lines = o3d.geometry.LineSet()
        for i in human_lines:
            all_human_lines += i

        if args.vis:
            def o3d_to_pv(geom):
                if isinstance(geom, o3d.geometry.TriangleMesh):
                    verts = np.asarray(geom.vertices)
                    faces = np.asarray(geom.triangles)
                    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
                    return pv.PolyData(verts, faces_pv)
                elif isinstance(geom, o3d.geometry.LineSet):
                    verts = np.asarray(geom.points)
                    lines = np.asarray(geom.lines)
                    lines_pv = np.hstack([np.full((lines.shape[0], 1), 2), lines]).astype(np.int64)
                    return pv.PolyData(verts, lines=lines_pv)
                elif isinstance(geom, o3d.geometry.PointCloud):
                    verts = np.asarray(geom.points)
                    return pv.PolyData(verts)
                else:
                    raise TypeError(f"Unsupported geometry type: {type(geom)}")

            pv_mesh_pr = o3d_to_pv(mesh_pr)
            pv_mesh = o3d_to_pv(mesh)
            pv_lines_optimized = o3d_to_pv(all_lines_optimized)
            pv_optimized_lines = o3d_to_pv(all_sparse_lines)
            pv_human_lines = o3d_to_pv(all_human_lines)

            plotter = pv.Plotter(shape=(2, 3), window_size=(1800, 1000))

            plotter.subplot(0, 0)
            plotter.add_text("Actual Mesh", font_size=10)
            plotter.add_mesh(pv_mesh_pr, color='lightgray', show_edges=True)

            plotter.subplot(0, 1)
            plotter.add_text("Curve Network", font_size=10)
            plotter.add_mesh(pv_lines_optimized, color='blue')

            plotter.subplot(0, 2)
            plotter.add_text("Sparse Network", font_size=10)
            plotter.add_mesh(pv_optimized_lines, color='green')

            plotter.subplot(1, 0)
            plotter.add_text("Decimated Mesh", font_size=10)
            plotter.add_mesh(pv_mesh, color='lightgray', show_edges=True)

            plotter.subplot(1, 1)
            plotter.add_text("Human Network", font_size=10)
            plotter.add_mesh(pv_human_lines, color='red')

            plotter.show()
            plotter.close()
            del plotter, pv_mesh, pv_lines_optimized, pv_optimized_lines, pv_human_lines, pv_mesh_pr

        if args.save:
            with h5py.File(save_path, "w") as f:
                grp = f.create_group("curve_network")
                grp.create_dataset("points", data=np.asarray(all_lines_optimized.points))
                grp.create_dataset("lines", data=np.asarray(all_lines_optimized.lines))

                chain_indices = []
                offset = 0
                for lineset in lines_optimized:
                    num_lines = np.asarray(lineset.lines).shape[0]
                    chain_indices.append(np.arange(offset, offset + num_lines))
                    offset += num_lines
                grp.create_dataset("chain_indices", data=np.array(chain_indices, dtype=object), dtype=h5py.vlen_dtype(np.int32))

                grp = f.create_group("sparse_network")
                grp.create_dataset("points", data=np.asarray(all_sparse_lines.points))
                grp.create_dataset("lines", data=np.asarray(all_sparse_lines.lines))

                chain_indices = []
                offset = 0
                for lineset in sparse_lines:
                    num_lines = np.asarray(lineset.lines).shape[0]
                    chain_indices.append(np.arange(offset, offset + num_lines))
                    offset += num_lines
                grp.create_dataset("chain_indices", data=np.array(chain_indices, dtype=object), dtype=h5py.vlen_dtype(np.int32))

                grp = f.create_group("human_network")
                grp.create_dataset("points", data=np.asarray(all_human_lines.points))
                grp.create_dataset("lines", data=np.asarray(all_human_lines.lines))

                chain_indices = []
                offset = 0
                for lineset in human_lines:
                    num_lines = np.asarray(lineset.lines).shape[0]
                    chain_indices.append(np.arange(offset, offset + num_lines))
                    offset += num_lines
                grp.create_dataset("chain_indices", data=np.array(chain_indices, dtype=object), dtype=h5py.vlen_dtype(np.int32))

    except Exception as e:
        print(f"Error processing file {path}: {e}")
        traceback.print_exc()
