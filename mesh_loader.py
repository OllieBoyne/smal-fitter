from pytorch3d.io import load_obj, save_obj
import os
import torch
from pytorch3d.structures import Meshes
from utils import cartesian_rotation, plot_pointcloud
import numpy as np

def load_target_meshes(mesh_dir, sorting = lambda arr: arr, n_meshes=None, frame_step=1, device="cuda:0"):
	"""Given a dir of meshes, loads all and returns mesh names, and meshes as Mesh object.

	sorting: function for sorting the files in the mesh_dir (by default does nothing)"""
	# load all meshes
	mesh_names = []
	all_verts, all_faces_idx = [], []

	obj_list = sorting(os.listdir(mesh_dir))
	if n_meshes is not None: obj_list = obj_list[:n_meshes]
	obj_list = obj_list[::frame_step]

	for obj_file in obj_list[:1]:
		mesh_names.append(obj_file[:-4]) # Get name of mesh
		target_obj = os.path.join(mesh_dir, obj_file)
		verts, faces, aux = load_obj(target_obj, load_textures=False) # Load mesh with no textures
		faces_idx = faces.verts_idx.to \
			(device) # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
		verts = verts.to(device) # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh

		# Center and scale for normalisation purposes
		centre = verts.mean(0)
		verts = verts - centre
		scale = max(verts.abs().max(0)[0])
		verts = verts /scale

		# ROTATE TARGET MESH TO GET IN DESIRED DIRECTION
		R1 = cartesian_rotation("z", np.pi/2)
		R2 = cartesian_rotation("y", np.pi/2)
		verts = torch.mm(verts, R1.T)
		verts = torch.mm(verts, R2.T)

		all_verts.append(verts), all_faces_idx.append(faces_idx)

	print(f"{len(all_verts)} target meshes loaded.")

	target_meshes = Meshes(verts=all_verts, faces=all_faces_idx) # All loaded target meshes together

	# from matplotlib import pyplot as plt
	# fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
	# plot_pointcloud(ax, target_meshes[0])
	# plt.show()

	return mesh_names, target_meshes