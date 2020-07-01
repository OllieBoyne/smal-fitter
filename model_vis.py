"""Visualise a generated SMAL type model, producing animations of changing key parameters"""

from smbld_model.config import SMPL_DATA_PATH, SMPL_MODEL_PATH, NEW_MODEL_PATH, NEW_DATA_PATH
from smbld_model.smbld_mesh import SMBLDMesh
import numpy as np
import torch

from matplotlib import pyplot as plt
from utils import plot_mesh, save_animation

models = {
	"default": dict(name="default smal", data_path=SMPL_DATA_PATH, model_path=SMPL_MODEL_PATH,
					shape_family_id=-1, num_betas = 20),
	"new": dict(name="new model", data_path=NEW_DATA_PATH, model_path=NEW_MODEL_PATH,
					shape_family_id=0, num_betas=13)
} ## Model name : SMBLDMesh kwargs

# Set the device
if torch.cuda.is_available():
	device = torch.device("cuda:0,1")

else:
	device = torch.device("cpu")

def vis_shape_params(name="default", num_betas=20, fps=15, **model_kwargs):
	"""Load SMBLD model. Wiggle each shape param in turn"""

	mesh = SMBLDMesh(**model_kwargs, num_betas=num_betas, device = device)

	fig, ax = plt.subplots(figsize=(10,10), subplot_kw={"projection": "3d"})
	trisurfs = []

	num_steps = N = 2 * fps
	# num_betas = num_betas

	## shape range goes from 0 -> 1 -> -1 -> 0 in equally spaced steps
	shape_range = np.concatenate([np.linspace(0, 1, num_steps//4), np.linspace(1, -1, num_steps//2), np.linspace(-1, 0, num_steps//4)])
	shape_range = np.pad(shape_range, (0, N - len(shape_range))) # pad to size N
	n_frames = num_steps * num_betas

	## plot blank mesh
	trisurfs = plot_mesh(ax, mesh.get_meshes(), zoom = 1.5, equalize=True) # adjust axes to initial mesh

	def anim(i):
		[t.remove() for t in trisurfs]  # clear existing meshes

		### update SMBLD
		cur_beta_idx = i // num_steps
		cur_step = i % num_steps
		val = shape_range[cur_step]

		mesh.multi_betas[0, cur_beta_idx] = val  # Update betas
		trisurfs[:] = plot_mesh(ax, mesh.get_meshes(), equalize=False) # get new mesh

		### update text
		fig.suptitle(f"{name.title()}\nS{cur_beta_idx} : {val:+.2f}", fontsize=30)

	ax.axis("off")
	fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
	save_animation(fig, anim, n_frames=n_frames, fmt="gif", title=f"{name}_betas", fps=fps)


if __name__ == "__main__":
	vis_shape_params(**models["default"], fps=15)
	vis_shape_params(**models["new"], fps=15)
