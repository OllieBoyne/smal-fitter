import numpy as np
from matplotlib import pyplot as plt
import os, torch
from pytorch_arap.pytorch_arap import arap_utils
from pytorch3d.structures import Meshes

def try_mkdir(loc):
	if os.path.isdir(loc):
		return None
	os.mkdir(loc)

def try_mkdirs(locs):
	for loc in locs: try_mkdir(loc)

def equal_3d_axes(ax, X, Y, Z, zoom=1.0):
	"""
	For pyplot 3D axis, sets all axes to same lengthscale through trick found here:
	https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to"""

	xmax, xmin, ymax, ymin, zmax, zmin = X.max(), X.min(), Y.max(), Y.min(), Z.max(), Z.min()

	max_range = np.array([xmax - xmin, ymax - ymin, zmax - zmin]).max() / (2.0 * zoom)

	mid_x = (xmax + xmin) * 0.5
	mid_y = (ymax + ymin) * 0.5
	mid_z = (zmax + zmin) * 0.5
	ax.set_xlim(mid_x - max_range, mid_x + max_range)
	ax.set_ylim(mid_y - max_range, mid_y + max_range)
	ax.set_zlim(mid_z - max_range, mid_z + max_range)

def plot_mesh(ax, mesh: Meshes, label="", colour="blue", equalize=True, zoom=1.5, alpha=1.0):
	"""Given a PyTorch Meshes object, plot the mesh on a 3D axis"""

	verts = mesh.verts_padded()
	faces = mesh.faces_padded()
	trisurfs = arap_utils.plot_meshes(ax, verts, faces, color=colour, change_lims=equalize, zoom=zoom, prop=False,
			alpha = alpha)
	ax.plot([], [], color=colour, label=label)

	return trisurfs

def plot_meshes(target_meshes, src_meshes, mesh_names=[], title="", figtitle="", out_dir="static_fits_output/pointclouds"):
	"""Plot and save fig of point clouds, with 3 figs side by side:
	[target mesh, src_mesh, both]"""

	for n in range(len(target_meshes)):
		fig = plt.figure(figsize=(15, 5))
		axes = [fig.add_subplot(int(f"13{n}"), projection="3d") for n in range(1, 4)]

		for ax in axes:
			ax.set_xlabel('x')
			ax.set_ylabel('y')
			ax.set_zlabel('z')

		colours = ["green", "blue"]
		labels = ["target", "SMAL"]
		for i, mesh in enumerate([target_meshes[n], src_meshes[n]]):
			for j, ax in enumerate([axes[1 + i == 1], axes[2]]):
				plot_mesh(ax, mesh, colour=colours[i], label=labels[i], alpha=[0.5, 1][j==0])

		fig.suptitle(figtitle)
		for ax in axes:
			ax.legend()

		if mesh_names == []:
			name = n
		else:
			name = mesh_names[n]

		try_mkdir(out_dir)

		plt.savefig(
			f"{out_dir}/{name} - {title}.png")  # ADD BETTER NAMING CONVENTION TODO CONSIDER MESH NAMES (PASS THIS TO STAGE OBJECT?)
		plt.close(fig)


def plot_pointcloud(ax, mesh, label="", colour="blue",
					equalize=True, zoom=1.5):
	"""Given a Meshes object, plots the mesh on ax (ax must have projection=3d).

	equalize = adjust axis limits such that all axes are equal"""

	verts = mesh.verts_packed()
	x, y, z = verts.clone().detach().cpu().unbind(1)

	s = ax.scatter3D(x, y, z, c=colour, label=label, alpha=0.3)

	if equalize:
		equal_3d_axes(ax, x, y, z, zoom=zoom)

	return s, (x, y, z)


def plot_pointclouds(target_meshes, src_meshes, mesh_names=[], title="", figtitle="", out_dir="static_fits_output/pointclouds"):
	"""Plot and save fig of point clouds, with 3 figs side by side:
	[target mesh, src_mesh, both]"""

	for n in range(len(target_meshes)):
		fig = plt.figure(figsize=(15, 5))
		axes = [fig.add_subplot(int(f"13{n}"), projection="3d") for n in range(1, 4)]

		for ax in axes:
			ax.set_xlabel('x')
			ax.set_ylabel('y')
			ax.set_zlabel('z')

		colours = ["green", "blue"]
		labels = ["target", "SMAL"]
		for i, mesh in enumerate([target_meshes[n], src_meshes[n]]):
			for ax in [axes[1 + i == 1], axes[2]]:
				plot_pointcloud(ax, mesh, colour=colours[i], label=labels[i])

		fig.suptitle(figtitle)
		for ax in axes:
			ax.legend()

		if mesh_names == []:
			name = n
		else:
			name = mesh_names[n]

		try_mkdir(out_dir)

		plt.savefig(
			f"{out_dir}/{name} - {title}.png")  # ADD BETTER NAMING CONVENTION TODO CONSIDER MESH NAMES (PASS THIS TO STAGE OBJECT?)
		plt.close(fig)

def cartesian_rotation(dim="x", rot=0):
	"""Given a cartesian direction of rotation, and a rotation in radians, returns pytorch rotation matrix"""

	i = "xyz".find(dim)
	R = torch.eye(3)
	if rot != 0:
		j, k = (i + 1) % 3, (i + 2) % 3  # other two of cyclic triplet
		R[j, j] = R[k, k] = np.cos(rot)
		R[j, k] = - np.sin(rot)
		R[k, j] = np.sin(rot)

	return R

def stack_as_batch(tensor: torch.Tensor, n_repeats=1, dim=0) -> torch.Tensor:
	"""Inserts new dim dimension, and stacks tensor n times along that dimension"""
	res = tensor.unsqueeze(dim)
	repeats = [1] * res.ndim
	repeats[dim] = n_repeats # repeat across target dimension
	res = res.repeat(*repeats)
	return res

def save_animation(fig, func, n_frames, fmt="gif", fps=15, title="output", callback=True, **kwargs):
	"""Save matplotlib animation."""

	arap_utils.save_animation(fig, func, n_frames, fmt="gif", fps=fps, title=title, callback=True, **kwargs )