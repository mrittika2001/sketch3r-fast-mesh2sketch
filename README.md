# Sketch3R Sketch Generation — Unofficial Implementation

> **This is an unofficial implementation.** It is not affiliated with or endorsed by the authors of the paper.

An unofficial implementation of the **mesh-to-sketch generation** stage of:

**Sketch3R: Rapid and Realistic 3D VR Sketch Creation to Shape Retrieval** — WACV 2026
[[Paper]](https://openaccess.thecvf.com/content/WACV2026/papers/Halder_Sketch3R_Rapid_and_Realistic_3D_VR_Sketch_Creation_to_Shape_WACV_2026_paper.pdf)

The repository covers Section 3.1 of the paper (generating human-like 3D VR sketches from meshes). The retrieval network (Section 3.2) is not included.

<p align="center"><i>3D mesh → quasi-planar patches → boxlets → surface-projected strokes → sparsified sketch → human-like sketch</i></p>

## What it does

Given a triangle mesh of any topology (non-manifold, multi-component and non-watertight meshes are fine), the pipeline produces a 3D line sketch that resembles one drawn by hand in VR:

1. **Normalise** the mesh into a unit cube and **decimate** it to roughly 3,000 faces.
2. **Partition** the surface into quasi-planar, spatially coherent patches by clustering faces on a joint feature of geodesic position and face normal. Disconnected clusters are split, and tiny clusters are merged into their nearest neighbour.
3. **Score each patch** with a curvature index that combines the area-normalised cotangent Laplacian with the patch's Euler characteristic (Gauss–Bonnet), and label patches as low or high curvature.
4. **Fit boxlets**: for each patch, take its convex hull, fit an oriented cuboid, snap the 8 corners to mesh vertices and use the cuboid's faces as strokes. High-curvature patches receive a subdivided boxlet.
5. **Project** densely sampled stroke points onto the patch surface using a signed distance field. This is the **machine sketch**.
6. **Sparsify** by removing strokes that are both near-parallel and spatially close to a longer stroke. This is the **sparse sketch**.
7. **Humanify** each stroke with smooth helical noise so that it looks hand-drawn. This is the **human sketch**.

All three sketches are written to a single HDF5 file.

## Repository layout

| File | Purpose |
|---|---|
| `main.py` | Process a single mesh (optional interactive visualisation) |
| `batch_pipeline.py` | Process every `.obj` in a dataset folder, resumable, with logging |
| `visualize_h5.py` | View a generated `.h5` next to the source mesh |
| `load_mesh.py` | Load and normalise a mesh to the unit cube |
| `meshlib_decim.py` | Mesh decimation |
| `create_graph_cluster.py` | Normal-angle segmentation used to choose the cluster count |
| `create_cluster_euler.py` | Patch clustering, refinement and curvature index |
| `optimize_bezier.py` | Boxlet fitting, surface projection, sparsification and humanification |
| `sdf_project.py`, `utils.py` | SDF and geometry helpers |
| `assets/` | Example meshes |

## Setup

Tested with **Python 3.10**.

### Linux with an NVIDIA GPU (recommended)

```bash
conda create -n sketch python=3.10 -y
conda activate sketch
pip install -r requirements.txt
```

`requirements.txt` targets a CUDA 12.x driver. For CUDA 11.x, replace `cupy-cuda12x` with `cupy-cuda11x` and install PyTorch from `https://download.pytorch.org/whl/cu118`.

### CPU only (including macOS on Apple Silicon)

The pipeline falls back to the CPU automatically, and CuPy is optional:

```bash
conda create -n sketch python=3.10 -y
conda activate sketch
grep -v cupy requirements.txt | grep -v '^torch' > requirements-cpu.txt
pip install -r requirements-cpu.txt
conda install -c conda-forge pytorch -y
```

On recent macOS versions, if importing `scipy` or `torch` fails with a `dlopen` or `OMP` error, replace the pip builds with the conda-forge ones:

```bash
pip uninstall -y scipy torch
conda install -c conda-forge scipy pytorch -y
```

## Running

### Single mesh

```bash
python main.py --mesh assets/painted_wooden_table.obj --out_root outputs --save
```

| Argument | Description |
|---|---|
| `--mesh` | Input mesh (any format Open3D can read: `.obj`, `.ply`, `.off`, `.stl`, …) |
| `--out_root` | Output root folder |
| `--save` | Write the sketches to HDF5 |
| `--vis` | Open an interactive viewer comparing the mesh and the three sketches |

The output is written to `<out_root>/<parent_folder>/<mesh_name>_all_output.h5`, e.g. `outputs/assets/painted_wooden_table_all_output.h5`. Meshes whose output already exists are skipped.

### Whole dataset

```bash
python batch_pipeline.py --dataset /path/to/dataset --save_root outputs
```

Every `.obj` under `--dataset` is processed. Progress is logged to `logs/` and the file list is cached in `cache/`, so an interrupted run resumes where it stopped. To select a GPU, set `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=1 python batch_pipeline.py --dataset /path/to/ModelNet40 --save_root outputs
```

The decimation step writes a temporary `tmp.ply` to the working directory, so run parallel jobs from separate directories.

### Visualising a result

```bash
python visualize_h5.py outputs/assets/painted_wooden_table_all_output.h5 --mesh assets/painted_wooden_table.obj
```

### Output format

Each `.h5` file contains three groups — `curve_network` (machine sketch), `sparse_network` (sparsified sketch) and `human_network` (human-like sketch) — each with:

| Dataset | Shape | Content |
|---|---|---|
| `points` | `(P, 3)` | Stroke vertices in the unit-cube frame |
| `lines` | `(L, 2)` | Line segments as indices into `points` |
| `chain_indices` | `(S,)` variable-length | For each stroke, the indices of its rows in `lines` |

```python
import h5py
with h5py.File("outputs/assets/painted_wooden_table_all_output.h5") as f:
    pts   = f["human_network/points"][:]
    lines = f["human_network/lines"][:]
    strokes = [lines[idx] for idx in f["human_network/chain_indices"]]
```

## Improvements over the paper

This implementation follows the structure of the paper's sketch generation pipeline, with the following changes, mostly made for speed and robustness on large datasets. Together they reduce generation time from **about 30 s to about 5 s per sketch** (the paper reports 30 s – 1 min). Small, simple meshes take under a second.

**Pre-processing**
- **Mesh decimation.** Inputs are reduced to about 3,000 faces (bounded error) before clustering, so runtime no longer grows with the resolution of the input mesh.

**Patch partitioning**
- **Landmark geodesics instead of all-pairs geodesics.** Rather than an all-pairs face-to-face geodesic matrix (hybrid Dijkstra / Bellman–Ford), geodesic distances are computed from 128 randomly chosen landmark faces. Cost drops from roughly O(|F|²) to O(L·|F| log |F|), and the full |F|×|F| matrix is never stored.
- **GPU k-means instead of spectral clustering.** Faces are clustered with a vectorised GPU k-means on the joint feature `[0.3 · normalised geodesic, 1.0 · face normal]`. This avoids building a dense affinity matrix and its eigendecomposition.
- **Automatic cluster count.** The number of clusters is the larger of a minimum (10) and the number of normal-coherent segments found by a fast angle-based pre-segmentation, so no per-mesh tuning is needed.
- **Bounded, scalable refinement.** After disconnected clusters are split, small clusters (fewer than 3 vertices) are merged into their nearest neighbour, and the total is capped at 300 patches. For very large cluster counts, a hierarchical spatial-grid merge reduces cost from O(N²) to roughly O(N log N).

**Curvature and boxlets**
- **Adaptive curvature threshold.** The curvature index is min–max normalised per mesh and split with **Otsu's method** instead of a fixed `δ_curve = 0.5`, so the low/high split adapts to each shape.
- **Uniform boxlet subdivision.** High-curvature patches have **every** face of their cuboid split at edge midpoints (12 quads), rather than only the longest side, which captures more structure on curved parts.
- **Robust boxlet fitting.** Vertices receive a small, seeded offset along their normals before the convex hull is computed. This prevents degenerate hulls on perfectly planar patches; the offset is removed again before projection. There is also a fallback for patches where the cuboid fit fails.
- **Batched SDF projection.** All stroke samples of a patch are projected in one batched signed-distance query: one centre evaluation plus six finite-difference evaluations for the gradient. CuPy is used for large batches.

**Sparsification and humanification**
- **Length-aware sparsification.** Strokes are processed from longest to shortest, so when two strokes are redundant, the longer one is kept. The angular test uses `|cos θ| > 0.8` with a centroid distance below 0.05, which removes near-parallel duplicates more aggressively and yields cleaner, sparser sketches.
- **Deterministic helical noise.** Each stroke is perturbed in windows of 10–30 points along a helix in its local normal/binormal frame, with a triangular envelope that keeps window boundaries and stroke endpoints fixed. Noise is seeded per stroke, so outputs are reproducible.

**Engineering**
- Batch processing with resume support, per-stage timing logs and HDF5 output that stores all three sketch levels together.
- Automatic CPU fallback when no CUDA device is available.

## Example meshes

| File | Source | Licence |
|---|---|---|
| `assets/stanford-bunny.obj` | [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/) | Free for research use |
| `assets/painted_wooden_table.obj` | [Poly Haven — Painted Wooden Table](https://polyhaven.com/a/painted_wooden_table) (geometry only) | CC0 |

## License

This code is released under the [MIT License](LICENSE). The example meshes and third-party dependencies are covered by their own licences.

## Third-party licences

Mesh decimation uses [MeshLib](https://meshlib.io), which is free for non-commercial use only. Check its licence before any commercial use, or replace `meshlib_decim.py` with another decimator, such as Open3D's `simplify_quadric_decimation`.

## Citation

If you use this code, please cite the original paper:

```bibtex
@InProceedings{Halder_2026_WACV,
    author    = {Halder, Mritunjoy and Shukla, Shivam Ashok and Tiwari, Lokender and Mittal, Raghav and Bhowmick, Brojeshwar},
    title     = {Sketch3R: Rapid and Realistic 3D VR Sketch Creation to Shape Retrieval},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {March},
    year      = {2026},
    pages     = {8680-8689}
}
```
