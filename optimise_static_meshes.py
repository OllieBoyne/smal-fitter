"""Optimise the SMBLD model to a series of .obj files"""

"""Using the method described in https://pytorch3d.org/tutorials/deform_source_mesh_to_target_meshes,

Attempts to optimise the SMAL model such that the mesh best fits with the input mesh"""

import sys, os

sys.path.append(os.path.dirname(sys.path[0]))
from utils import plot_pointclouds, try_mkdir, try_mkdirs
import torch
from mesh_loader import load_target_meshes
from pytorch3d.structures import Meshes
from smbld_model.smbld_mesh import SMBLDMesh
from optimiser import Stage

# Set the device
if torch.cuda.is_available():
	device = torch.device("cuda:0,1")

device = torch.device("cpu")

targ_dirs = ["static_meshes", "static_fits_output"]
try_mkdirs(targ_dirs) # produce targ dirs


def optimise_to_static_meshes():
	"""Individually optimise the SMAL model to 13 different target meshes, through a SMAL parameter optimisation stage and a vertex deformation stage. Save outputs as numpy arrays."""

	mesh_names, target_meshes = load_target_meshes(mesh_dir=r"static_meshes", device=device)

	n_batch = batch_size = len(target_meshes)  # Size of all meshes in batch

	SMBLD = SMBLDMesh(n_batch=n_batch, device=device)


	# Load and plot initial SMAL Mesh
	verts, faces_idx = SMBLD.get_verts()
	src_mesh = Meshes(verts, faces_idx)

	plot_pointclouds(target_meshes, src_mesh, mesh_names=mesh_names, title="0 - Init")

	# Other parameters to optimise - vertex deform
	deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device,
							  requires_grad=True)  # Tensor to optimise

	stage_kwaargs = {
		"mesh_names": mesh_names,
		"target_meshes": target_meshes,
		"loss_weights": dict(w_laplacian = 0)
	}

	nits = 1
	stage1 = Stage(nits, SMBLD.smbld_params, SMBLD, name="1 - Initial fit", lr=1e-1, **stage_kwaargs)

	stage2 = Stage(nits, SMBLD.smbld_params, SMBLD, name="2 - Refine", lr=5e-2, **stage_kwaargs)

	stage3 = Stage(nits, SMBLD.smbld_params + SMBLD.deform_params, SMBLD, name="3 - Deform",
				   lr=5e-2, **stage_kwaargs)

	stages = [stage1]#, stage2, stage3]

	out_dir = r"static_fits_output"

	for n, stage in enumerate(stages):
		plot = n==len(stages) - 1 # plot only if on final stage
		stage.run(plot = plot)
		# stage.plot_losses(f"/data/cvfs/ob312/smal_fitter/fit_to_mesh_outputs/losses-{stage.name}.png")

	# Save all SMAL params
	try_mkdir(out_dir)
	SMBLD.save_npz(out_dir)


if __name__ == "__main__":
	optimise_to_static_meshes()
