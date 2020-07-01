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


def optimise_to_static_meshes(arap = True):
	"""Individually optimise the SMAL model to 13 different target meshes, through a SMAL parameter optimisation stage and a vertex deformation stage. Save outputs as numpy arrays."""

	mesh_names, target_meshes = load_target_meshes(mesh_dir=r"static_meshes", device=device)

	## only take first mesh for now
	# mesh_names, target_meshes = mesh_names[:1], target_meshes[:1]

	n_batch = batch_size = len(target_meshes)  # Size of all meshes in batch

	SMBLD = SMBLDMesh(n_batch=n_batch, device=device, shape_family_id=-1) # use no shape family

	# Load and plot initial SMAL Mesh
	verts, faces_idx = SMBLD.get_verts()
	src_mesh = Meshes(verts, faces_idx)

	# plot_meshes(target_meshes, src_mesh, mesh_names=mesh_names, title="0 - Init", figtitle="Initialisation")

	out_dir = r"static_fits_output"

	stage_kwaargs = {
		"mesh_names": mesh_names,
		"target_meshes": target_meshes,
	}

	deform_weights = dict(
		w_laplacian=0, w_arap=0.000, w_normal=0.00, w_edge=0.00,
	)

	if arap: 
		deform_weights["w_arap"] = 0.01
		deform_weights["w_normal"] = 0.002
		# deform_weights["w_laplacian"] = 0.001

	manager = StageManager(out_dir=out_dir)

	nits = 1000
	manager.add_stage( Stage(100, SMBLD.smbld_params, SMBLD, name="1 - Initial fit", lr=1e-1, **stage_kwaargs) )
	manager.add_stage( Stage(nits, SMBLD.smbld_params, SMBLD, name="2 - Refine", lr=1e-3,  **stage_kwaargs) )
	name = "3 - deform arap" if arap else "3 - deform"
	manager.add_stage( Stage(200, SMBLD.deform_params, SMBLD, name=name, loss_weights=deform_weights,
				   lr=2e-2, **stage_kwaargs) )

	manager.run()
	manager.plot_losses()

	print(SMBLD.deform_verts.mean(), SMBLD.deform_verts.sum())

	# Save all SMAL params
	try_mkdir(out_dir)
	SMBLD.save_npz(out_dir, title=f"{['', 'arap'][arap]}")


if __name__ == "__main__":
	# optimise_to_static_meshes(arap = False)
	optimise_to_static_meshes(arap = True)
