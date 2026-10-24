import h5py
import numpy as np
import open3d as o3d

def read_and_visualize_separate(save_path):
    with h5py.File(save_path, "r") as f:
        networks = {}

        def load_lineset_group_separate(grp):
            points = np.array(grp["points"])
            lines = np.array(grp["lines"])
            chain_indices = grp["chain_indices"]

            line_sets = []
            for idx_lines in chain_indices:
                idx_lines = idx_lines[()]
                ls = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(points)
                ls.lines = o3d.utility.Vector2iVector(lines[idx_lines])
                line_sets.append(ls)
            return line_sets

        curve_ls_list = load_lineset_group_separate(f["curve_network"])
        networks["curve_network"] = curve_ls_list

        sparse_ls_list = load_lineset_group_separate(f["sparse_network"])
        networks["sparse_network"] = sparse_ls_list

        human_ls_list = load_lineset_group_separate(f["human_network"])
        networks["human_network"] = human_ls_list

        mesh_grp = f["stylized_human_network"]
        vertices = np.array(mesh_grp["vertices"])
        faces = np.array(mesh_grp["faces"])
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        if "colors" in mesh_grp:
            colors = np.array(mesh_grp["colors"])
            if colors.shape[0] == len(vertices):
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            elif colors.shape[0] == len(faces):
                mesh.vertex_colors = o3d.utility.Vector3dVector(
                    np.repeat(colors, 3, axis=0)[:len(vertices)]
                )

    for name, ls_list in networks.items():
        print(f"Visualizing {name} ({len(ls_list)} chains)")
        o3d.visualization.draw_geometries(ls_list)

    o3d.visualization.draw_geometries([mesh])

    return networks

if __name__ == "__main__":
    import sys
    read_and_visualize_separate(sys.argv[1])
