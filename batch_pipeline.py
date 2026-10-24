#!/usr/bin/env python3
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import gc
import sys
import h5py
import time
import torch
import argparse
import traceback
import numpy as np
import open3d as o3d
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


from load_mesh import load_mesh_and_fit_unit
from create_cluster_euler import return_cluster
from create_graph_cluster import segment_mesh_by_angle
from optimize_bezier import optimize_curve, post_process_curves, humanify
from meshlib_decim import decimate


def log(msg, log_file):
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")


def discover_files(dataset_dir, cache_file, log_file):
    if cache_file.exists():
        log("Resuming: Loading file list from cache.", log_file)
        files = cache_file.read_text().splitlines()
    else:
        log("First run: Scanning filesystem for .obj files...", log_file)
        files = sorted(str(p) for p in Path(dataset_dir).rglob("*.obj"))
        cache_file.write_text("\n".join(files))
        log("Cache created.", log_file)
    return [f for f in files if f.strip()]


def load_completed(completed_log):
    if completed_log.exists():
        return set(completed_log.read_text().splitlines())
    return set()


def get_save_path(obj_file, save_root):
    base = ''.join(map(str, obj_file.split('.')[:-1])) + '_all_output.h5'
    return os.path.join(str(save_root), '/'.join(base.split("/")[-2:]))


def save_results(save_path, lines_optimized, sparse_lines, human_lines):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def merge_linesets(linesets):
        merged = o3d.geometry.LineSet()
        for ls in linesets:
            merged += ls
        return merged

    def write_network(f, group_name, merged, linesets):
        grp = f.create_group(group_name)
        grp.create_dataset("points", data=np.asarray(merged.points))
        grp.create_dataset("lines", data=np.asarray(merged.lines))
        chain_indices = []
        offset = 0
        for ls in linesets:
            num_lines = np.asarray(ls.lines).shape[0]
            chain_indices.append(np.arange(offset, offset + num_lines))
            offset += num_lines
        grp.create_dataset("chain_indices",
                           data=np.array(chain_indices, dtype=object),
                           dtype=h5py.vlen_dtype(np.int32))

    all_optimized = merge_linesets(lines_optimized)
    all_sparse = merge_linesets(sparse_lines)
    all_human = merge_linesets(human_lines)

    with h5py.File(save_path, "w") as f:
        write_network(f, "curve_network", all_optimized, lines_optimized)
        write_network(f, "sparse_network", all_sparse, sparse_lines)
        write_network(f, "human_network", all_human, human_lines)


def process_single(obj_file, args, save_root, log_file):
    save_path = get_save_path(obj_file, save_root)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        return "skipped"

    total_start = time.perf_counter()
    name = Path(obj_file).name

    def log_time(label, start):
        elapsed = time.perf_counter() - start
        with open(log_file, "a") as f:
            f.write(f"[{name}] {label}: {elapsed:.3f}s\n")

    t = time.perf_counter()
    mesh_pr = load_mesh_and_fit_unit(obj_file, vis=False)
    log_time("Load + Normalize", t)

    t = time.perf_counter()
    target_faces = 3000
    num_v1, num_f1 = np.array(mesh_pr.vertices).shape[0], np.array(mesh_pr.triangles).shape[0]
    if num_v1 == 0 or num_f1 == 0:
        raise ValueError("Invalid mesh! No triangles or faces?!")
    mesh = decimate(mesh_pr, target_faces, max_error=0.01)
    num_v2, num_f2 = np.array(mesh.vertices).shape[0], np.array(mesh.triangles).shape[0]
    log_time(f"Decimation {num_v1}->{num_v2} ({num_v2/(num_v1 + 1):.3f}) {num_f1}->{num_f2} ({num_f2/(num_f1 + 1):.3f})", t)
    if num_f2 > 100000:
        mesh = decimate(mesh, target_faces, max_error=0.05)
        num_v2, num_f2 = np.array(mesh.vertices).shape[0], np.array(mesh.triangles).shape[0]
        log_time(f"Decimation {num_v1}->{num_v2} ({num_v2/(num_v1 + 1):.3f}) {num_f1}->{num_f2} ({num_f2/(num_f1 + 1):.3f})", t)
    if num_f2 > 100000:
        raise ValueError("Decimation Failed!")

    t = time.perf_counter()
    _, num_clusters = segment_mesh_by_angle(mesh)
    log_time("Segmentation", t)

    t = time.perf_counter()
    V, F, N, D, high_curv_indices, _ = return_cluster(
        np.array(mesh.vertices),
        np.array(mesh.triangles),
        max(args.min_clusters, num_clusters),
        vis=False
    )
    log_time("Cluster Extraction", t)

    t = time.perf_counter()
    lines_optimized = optimize_curve(V, F, N, D, high_curv_indices=high_curv_indices, vis=False)
    log_time("Curve Optimization", t)

    t = time.perf_counter()
    sparse_lines = post_process_curves(lines_optimized, vis=False)
    log_time("Post-processing", t)

    t = time.perf_counter()
    human_lines = humanify(sparse_lines, noise_scale=0.0, vis=False)
    log_time("Humanification", t)

    t = time.perf_counter()
    save_results(save_path, lines_optimized, sparse_lines, human_lines)
    log_time("Save HDF5", t)

    with open(log_file, "a") as f:
        f.write(f"[{name}] Total: {time.perf_counter() - total_start:.3f}s\n")
        f.write("-" * 80 + "\n")

    return "ok"


def parse_args():
    parser = argparse.ArgumentParser(description="Batch 3D sketch extraction pipeline")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset directory containing .obj files")
    parser.add_argument("--save_root", type=str, required=True,
                        help="Output root")
    parser.add_argument("--min_clusters", type=int, default=10,
                        help="Minimum clusters per mesh (default: 10)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="CUDA device (default: 0)")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_dir = Path(args.dataset).resolve()
    dataset_name = dataset_dir.name

    log_dir = Path("logs")
    cache_dir = Path("cache")
    log_file = log_dir / f"{dataset_name}.log"
    completed_log = log_dir / f"{dataset_name}_completed.log"
    cache_file = cache_dir / f"{dataset_name}.cache"
    save_root = Path(args.save_root)

    log_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    save_root.mkdir(parents=True, exist_ok=True)
    log_file.touch()
    completed_log.touch()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    log(f"--- Starting Pipeline at {datetime.now()} ---", log_file)
    log(f"Dataset: {dataset_dir} ({dataset_name})", log_file)
    log(f"Save root: {save_root}", log_file)

    all_files = discover_files(dataset_dir, cache_file, log_file)
    total = len(all_files)

    if total == 0:
        log("Error: No .obj files found. Exiting.", log_file)
        sys.exit(1)

    log(f"Found {total} total .obj files.", log_file)

    completed = load_completed(completed_log)
    todo = [f for f in all_files if f not in completed]
    log(f"Already completed: {total - len(todo)} | Remaining: {len(todo)}", log_file)

    failures = 0
    skipped = 0
    for obj_file in tqdm(todo, desc=f"Processing {dataset_name}", unit="mesh"):
        try:
            result = process_single(obj_file, args, save_root, log_file)
            gc.collect()
            torch.cuda.empty_cache()
            if result == "skipped":
                skipped += 1
            with open(completed_log, "a") as f:
                f.write(obj_file + "\n")
        except Exception as e:
            failures += 1
            log(f"ERROR: {Path(obj_file).name}: {e}", log_file)
            log(traceback.format_exc(), log_file)

    log(f"--- Pipeline Finished at {datetime.now()} ---", log_file)
    log(f"Summary: {len(todo) - failures - skipped} processed, {skipped} skipped (existed), {failures} failed", log_file)


if __name__ == "__main__":
    main()
