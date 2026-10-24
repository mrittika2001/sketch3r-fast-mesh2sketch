import numpy as np
import open3d as o3d
from meshlib import mrmeshpy as mm
from meshlib import mrmeshnumpy as mn

def decimate(mesh_o3d, target_faces=3000, max_error=0.05, num_jobs=64):
    mesh = mn.meshFromFacesVerts(np.asarray(mesh_o3d.triangles, dtype=np.int32),
                             np.asarray(mesh_o3d.vertices, dtype=np.float32))
    num_faces = mesh.topology.getValidFaces().count()
    mesh.packOptimally()
    settings = mm.DecimateSettings()
    settings.maxDeletedFaces = max(0, num_faces - target_faces)
    settings.maxError = max_error
    settings.subdivideParts = num_jobs


    mm.decimateMesh(mesh, settings)
    mm.saveMesh(mesh, "tmp.ply")
    dec_mesh = o3d.io.read_triangle_mesh("tmp.ply")

    return dec_mesh
