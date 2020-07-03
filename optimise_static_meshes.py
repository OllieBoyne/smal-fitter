"""Optimise the SMBLD model to a series of .obj files"""

"""Using the method described in https://pytorch3d.org/tutorials/deform_source_mesh_to_target_meshes,

Attempts to optimise the SMAL model such that the mesh best fits with the input mesh"""

import sys, os

sys.path.append(os.path.dirname(sys.path[0]))
from utils import plot_pointclouds, try_mkdir, try_mkdirs, plot_meshes
import torch
from mesh_loader import load_target_meshes
from pytorch3d.structures import Meshes
from smbld_model.smbld_mesh import SMBLDMesh

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

	mesh_names, target_meshes = load_target_meshes(mesh_dir=r"static_meshes", device=device, n_meshes=n_meshes)
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

	deform_weights["w_arap"] = 0.001
	# deform_weights["w_normal"] = 0.02
	# deform_weights["w_laplacian"] = 0.001

	manager = StageManager(out_dir=out_dir)

	if method == "parameters":
		verts, faces_idx = SMBLD.get_verts()
		src_mesh = Meshes(verts, faces_idx)
		plot_meshes(target_meshes, src_mesh, mesh_names=mesh_names, title="0 - Init", figtitle="Initialisation") 	# Load and plot initial SMAL Mesh
		
		manager.add_stage( Stage(1000, "smbld", SMBLD, name="1 - Initialise", lr=3e-2, custom_lrs={"joint_rot": 5e-3}, **stage_kwaargs) )
		manager.add_stage( Stage(1000, "smbld", SMBLD, name="2 - Shape & pose", lr=2e-2, custom_lrs={"joint_rot": 5e-3},  **stage_kwaargs) )

	elif method == "deform":
		SMBLD.load_from_npz("static_fits_output/smbld_params_13_parameters.npz") # load from parameters stage

		manager.add_stage( Stage(1500, "deform", SMBLD, name="3 - deform arap", loss_weights=deform_weights,
					   lr=3e-3, **stage_kwaargs) )

	manager.run()
	# manager.plot_losses()

	SMBLD.save_npz(out_dir, title=f"{n_batch}_{method}") 	# Save all SMAL params


if __name__ == "__main__":
	n_meshes = None # if None, select all available meshes
	optimise_to_static_meshes(method="deform", n_meshes=n_meshes)
