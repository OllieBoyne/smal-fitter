"""Visualise a generated SMAL type model, producing animations of changing key parameters"""

from smbld_model.config import SMPL_DATA_PATH, SMPL_MODEL_PATH, NEW_MODEL_PATH, NEW_DATA_PATH
from smbld_model.smbld_mesh import SMBLDMesh
import numpy as np
import torch

from matplotlib import pyplot as plt
from utils import plot_mesh, save_animation, animator

models = {
	"default": dict(name="default smal", data_path=SMPL_DATA_PATH, model_path=SMPL_MODEL_PATH, shape_family_id=-1,
					num_betas=20),
	"new": dict(name="new model", data_path=NEW_DATA_PATH, model_path=NEW_MODEL_PATH, shape_family_id=0, num_betas=18)
}  # Model name : SMBLDMesh kwargs

# Set the device
if torch.cuda.is_available():
	device = torch.device("cuda:0,1")
else:
	device = torch.device("cpu")


def vis_shape_params(name="default", num_betas=20, fps=15, **model_kwargs):
	"""Load SMBLD model. Wiggle each shape param in turn"""

	mesh = SMBLDMesh(**model_kwargs, num_betas=num_betas, device=device)
	fig, ax = plt.subplots(figsize=(10, 10), dpi=30, subplot_kw={"projection": "3d"})
	num_steps = N = 2 * fps

	## shape range goes from 0 -> 1 -> -1 -> 0 in equally spaced steps
	shape_range = np.concatenate(
		[np.linspace(0, 1, num_steps // 4), np.linspace(1, -1, num_steps // 2), np.linspace(-1, 0, num_steps // 4)])
	shape_range = np.pad(shape_range, (0, N - len(shape_range)))  # pad to size N
	n_frames = num_steps * num_betas

	plot_mesh(ax, mesh.get_meshes(), zoom=1.5, equalize=True)  # plot blank mesh with axes equalized

	@animator(ax)
	def anim(i):
		# update SMBLD
		cur_beta_idx, cur_step = i // num_steps, i % num_steps
		val = shape_range[cur_step]
		mesh.multi_betas[0, cur_beta_idx] = val  # Update betas
		fig.suptitle(f"{name.title()}\nS{cur_beta_idx} : {val:+.2f}", fontsize=50)  # update text

		return dict(mesh=mesh.get_meshes(), equalize=False)

	ax.axis("off")
	fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
	save_animation(fig, anim, n_frames=n_frames, fmt="gif", title=f"{name}_betas", fps=fps)


def vis_pose_params(data_loc, name="default", num_betas=20, fps=15, **model_kwargs):
	"""Given a .npz file which contains N pose params, visualise the old SMBLD model with those pose params"""

	n_frames = np.load(data_loc)["joint_rot"].shape[0]

	mesh = SMBLDMesh(**model_kwargs, num_betas=num_betas, device=device, n_batch=n_frames)
	mesh.load_from_npz(data_loc)
	mesh_list = mesh.get_meshes()

	fig, ax = plt.subplots(figsize=(10, 10), dpi=30, subplot_kw={"projection": "3d"})
	plot_mesh(ax, mesh_list[0], zoom=1.5, equalize=True)  # plot blank mesh with axes equalized

	@animator(ax)
	def anim(i):
		return dict(mesh=mesh_list[i], equalize=False)

	ax.axis("off")
	fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
	save_animation(fig, anim, n_frames=n_frames, fmt="gif", title=f"{name}_pose", fps=fps)


if __name__ == "__main__":
	# vis_shape_params(**models["default"], fps=15)
	# vis_shape_params(**models["new"], fps=15)

	vis_pose_params(**models['default'], data_loc=r"animated_fits_output\smbld_params_fila_pose.npz")
