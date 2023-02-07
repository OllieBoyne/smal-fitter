"""Optimise the SMBLD model to a series of .obj files"""

"""Using the method described in https://pytorch3d.org/tutorials/deform_source_mesh_to_target_meshes,

Attempts to optimise the SMAL model such that the mesh best fits with the input mesh"""

import sys, os
joinp = os.path.join

sys.path.append(os.path.dirname(sys.path[0]))
from vis import plot_pointclouds, try_mkdir, try_mkdirs, plot_meshes, load_unity_meshes
import torch
nn = torch.nn

from pytorch3d.structures import Meshes
from smbld_model.smbld_mesh import SMBLDMesh
import numpy as np
from optimiser import Stage, StageManager

# Set the device
if torch.cuda.is_available():
	device = torch.device("cuda:0")

else:
	device = torch.device("cpu")

targ_dirs = ["static_meshes", "static_fits_output"]
try_mkdirs(targ_dirs) # produce targ dirs


def optimise_to_static_meshes(method="parameters", n_meshes=None):
	"""Individually optimise the SMAL model to 13 different target meshes, through a SMAL parameter optimisation stage and a vertex deformation stage. Save outputs as numpy arrays."""

	mesh_names, target_meshes = load_unity_meshes(mesh_dir=r"static_meshes", device=device, n_meshes=n_meshes)
	print(mesh_names)
	raise ValueError
	n_batch = batch_size = len(target_meshes)  # Size of all meshes in batch

	SMBLD = SMBLDMesh(n_batch=n_batch, device=device, shape_family_id=-1) # use no shape family

	out_dir = r"static_fits_output"
	try_mkdir(out_dir)

	stage_kwaargs = {
		"mesh_names": mesh_names,
		"target_meshes": target_meshes,
	}

	deform_weights = dict(
		w_laplacian=0, w_arap=0.000, w_normal=0.00, w_edge=0.00,
	)

	deform_weights["w_arap"] = 1e-4
	# deform_weights["w_normal"] = 0.02
	# deform_weights["w_laplacian"] = 0.001

	manager = StageManager(out_dir=out_dir)

	if method == "parameters":
		verts, faces_idx = SMBLD.get_verts()
		src_mesh = Meshes(verts, faces_idx)
		plot_meshes(target_meshes, src_mesh, mesh_names=mesh_names, title="0 - Init", figtitle="Initialisation") 	# Load and plot initial SMAL Mesh
		
		manager.add_stage( Stage(500, "smbld", SMBLD, name="1 - Initialise", lr=3e-2, custom_lrs={"joint_rot": 5e-3}, **stage_kwaargs) )
		manager.add_stage( Stage(2000, "smbld", SMBLD, name="2 - Shape & pose", lr=2e-2, custom_lrs={"joint_rot": 5e-3},  **stage_kwaargs) )

		manager.run()
		manager.plot_losses(out_src="parameters_losses")

	elif method == "deform":
		SMBLD.load_from_npz(f"static_fits_output/smbld_params_{n_batch}_parameters.npz") # load from parameters stage

		manager.add_stage( Stage(1500, "deform", SMBLD, name="3 - deform arap", loss_weights=deform_weights,
					   lr=5e-3, lr_decay=0.999, **stage_kwaargs) )

		manager.run()
		manager.plot_losses(out_src="deform_losses")

	SMBLD.save_npz(out_dir, title=f"{n_batch}_{method}") 	# Save all SMAL params

def optimise_to_animated_meshes(name="fila", method = "shape", n_meshes=5, frame_step = 50,
							plot = True):
	"""Given a sequence of .obj files for the same dog in several poses,
	optimise all SMBLD parameters to the first frame, then vertex deforms to the first frame,
	then optimise pose parameters to all frames.
	
	Currently only takes first fitted mesh from 'shape' stage through to 'pose' stage
	"""

	sort_func = lambda arr: sorted(arr, key = lambda i: int(i.split(".")[0])) # sort by num in file name

	mesh_names, target_meshes = load_unity_meshes(mesh_dir=f"animated_meshes/{name}", device=device, n_meshes=n_meshes, frame_step=frame_step,
	sorting = sort_func)
	n_batch = batch_size = len(target_meshes)  # Size of all meshes in batch

	SMBLD = SMBLDMesh(n_batch=n_batch, device=device, shape_family_id=-1,
	fixed_betas=True) # use no shape family

	out_dir = r"animated_fits_outputs"
	try_mkdir(out_dir)

	stage_kwaargs = dict(mesh_names=mesh_names, target_meshes=target_meshes, out_dir=out_dir)

	manager = StageManager(out_dir=out_dir)

	if method == "shape":
		verts, faces_idx = SMBLD.get_verts()
		src_mesh = Meshes(verts, faces_idx)
		plot_meshes(target_meshes, src_mesh, mesh_names=mesh_names, title="0 - Init", figtitle="Initialisation",
		out_dir = "animated_fits_outputs/meshes") 	# Load and plot initial SMAL Mesh
			
		manager.add_stage( Stage(500, "smbld", SMBLD, name="1 - Initialise", lr=3e-2, custom_lrs={"joint_rot": 5e-3}, **stage_kwaargs) )
		manager.add_stage( Stage(2000, "smbld", SMBLD, name="2 - Shape & pose", lr=2e-2, custom_lrs={"joint_rot": 5e-3},  **stage_kwaargs) )
		manager.add_stage( Stage(1500, "deform", SMBLD, name="3 - deform arap", loss_weights={"w_arap":1e-4},
						lr=5e-3, lr_decay=0.999, **stage_kwaargs) )

	if method == "pose":
		## Load shape method results
		init_data = np.load(joinp(out_dir, f"smbld_params_{name}_shape.npz"))
		multi_betas, deform_verts = init_data["multi_betas"][:1], init_data["deform_verts"][:1]

		## Assign to new model
		SMBLD.multi_betas = nn.Parameter(torch.from_numpy(multi_betas).repeat(n_batch, 1).to(device)) # Assign learned parameters to new model
		SMBLD.deform_verts = nn.Parameter(torch.from_numpy(deform_verts).repeat(n_batch, 1, 1).to(device))

		manager.add_stage( Stage(1000, "pose", SMBLD, name="4 - pose",
						lr=5e-3, lr_decay=1.0, **stage_kwaargs) )


	manager.run()
	manager.plot_losses(out_src=f"losses_{method}")

	SMBLD.save_npz(out_dir, title=f"{name}_{method}") 	# Save all SMAL params

if __name__ == "__main__":
	# n_meshes = None # if None, select all available meshes
	# optimise_to_static_meshes(method="deform", n_meshes=n_meshes)

	# optimise_to_animated_meshes(method="shape", n_meshes=1)
	optimise_to_animated_meshes(method="pose", n_meshes=5, frame_step=50, plot=True)
