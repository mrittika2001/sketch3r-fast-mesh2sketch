#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
import pyvista as pv
import open3d as o3d
from meshlib_decim import decimate
from load_mesh import load_mesh_and_fit_unit

def load_network(f, group_name):
    grp = f[group_name]
    points = grp["points"][:]
    lines = grp["lines"][:]
    lines_pv = np.hstack([np.full((lines.shape[0], 1), 2), lines]).astype(np.int64)
    return pv.PolyData(points, lines=lines_pv)


def main():
    parser = argparse.ArgumentParser(description="Visualize .h5 sketch output")
    parser.add_argument("file", type=str, help="Path to .h5 file")
    parser.add_argument("--mesh", type=str, help="Path to mesh file")
    args = parser.parse_args()

    with h5py.File(args.file, "r") as f:
        curve_network = load_network(f, "curve_network")
        sparse_network = load_network(f, "sparse_network")
        human_network = load_network(f, "human_network")

    geom = load_mesh_and_fit_unit(args.mesh, vis=False)
    verts = np.asarray(geom.vertices)
    faces = np.asarray(geom.triangles)
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
    mesh = pv.PolyData(verts, faces_pv)

    geom = decimate(geom, max_error=0.01)
    verts = np.asarray(geom.vertices)
    faces = np.asarray(geom.triangles)
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
    mesh_dec = pv.PolyData(verts, faces_pv)

    plotter = pv.Plotter(shape=(2, 3), window_size=(1800, 500))

    plotter.subplot(0, 0)
    plotter.add_text("Actual Mesh", font_size=10)
    plotter.add_mesh(mesh, color="lightgray", show_edges=True)

    plotter.subplot(0, 1)
    plotter.add_text("Decimated Mesh", font_size=10)
    plotter.add_mesh(mesh_dec, color="lightgray", show_edges=True)

    plotter.subplot(0, 2)
    plotter.add_text("Curve Network", font_size=10)
    plotter.add_mesh(curve_network, color="blue", line_width=2)

    plotter.subplot(1, 0)
    plotter.add_text("Sparse Network", font_size=10)
    plotter.add_mesh(sparse_network, color="green", line_width=2)

    plotter.subplot(1, 1)
    plotter.add_text("Human Network", font_size=10)
    plotter.add_mesh(human_network, color="red", line_width=2)

    plotter.link_views()
    plotter.show()

if __name__ == "__main__":
    main()
