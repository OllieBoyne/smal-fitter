"""Introduces Stage class - representing a Stage of optimising a batch of SMBLD meshes to target meshes"""

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.structures import Meshes
from pytorch_arap.pytorch_arap.arap import compute_energy as arap_loss
from pytorch_arap.pytorch_arap.arap import ARAPMeshes
from pytorch_arap.pytorch_arap.arap_utils import profile_backwards, time_function, Timer

from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from vis import plot_pointclouds, plot_meshes
import numpy as np
import os
from smbld_model.smbld_mesh import SMBLDMesh
from time import perf_counter
import gc

# default_weights = dict(w_chamfer=2.0, w_edge=1.0, w_normal=0.01, w_laplacian=0.1, w_scale=0.001, w_arap=0.001)
default_weights = dict(w_chamfer=1.0, w_edge=0, w_normal=0, w_laplacian=0, w_arap=0)

## Want to vary learning ratios between parameters, 
default_lr_ratios = []

class StageManager:
	"""Container for multiple stages of optimisation"""

	def __init__(self, out_dir = "static_fits_output"):
		self.stages = []
		self.out_dir = out_dir

	def run(self):
		for n, stage in enumerate(self.stages):
			stage.run(plot = True)

	def plot_losses(self, out_src = "losses"):
		"""Plot combined losses for all stages."""

		fig, ax = plt.subplots()
		it_start = 0 # track number of its
		for stage in self.stages:
			n_it = stage.n_it
			ax.semilogy(np.arange(it_start, it_start+n_it), stage.losses_to_plot, label=stage.name)
			it_start += n_it

		ax.legend()
		out_src = os.path.join(self.out_dir, out_src+".png")
		fig.savefig(out_src)
		plt.close(fig)

	def add_stage(self, stage):
		self.stages.append(stage)


class SMBLDMeshParamGroup:
	"""Object building on model.parameters, with modifications such as variable learning rate"""
	param_map = {
		"shape": ["global_rot", "trans", "multi_betas"],
		"smbld": ["global_rot", "joint_rot", "trans", "multi_betas"],
		"deform": ["deform_verts"],
		"pose": ["global_rot", "trans", "joint_rot"]
	}	# map of param_type : all attributes in SMBLDMesh used in optim

	def __init__(self, model, group="smbld", lrs = None):
		"""
		:param lrs: dict of param_name : custom learning rate
		"""

		self.model = model

		self.group = group
		assert group in self.param_map, f"Group {group} not in list of available params: {list(self.param_map.keys())}"

		self.lrs = {}
		if lrs is not None:
			for k, lr in lrs.items():
				self.lrs[k] = lr

	
	def __iter__(self):
		"""Return iterable list of all parameters"""
		out = []

		for param_name in self.param_map[self.group]:
			param = [getattr(self.model, param_name)]
			d = {"params": param}
			if param_name in self.lrs:
				d["lr"] = self.lrs[param_name]

			out.append(d)

		return iter(out)


class Stage:
	"""Defines a stage of optimisation, the optimisation parameters for the stage, ..."""

	def __init__(self, n_it: int, param_group: str, SMBLD: SMBLDMesh, target_meshes: Meshes, mesh_names=[], name="optimise",
				 loss_weights=None, lr=1e-3, lr_decay=1.0, out_dir="static_fits_output",
				 custom_lrs = None):
		"""
		n_its = integer, number of iterations in stage
		parameters = list of items over which to be optimised
		get_mesh = function that returns Mesh object for identifying losses
		name = name of stage

		lr_decay = factor by which lr decreases at each it"""


		self.n_it = n_it
		self.name = name
		self.out_dir = out_dir
		self.target_meshes = target_meshes
		self.mesh_names = mesh_names
		self.SMBLD = SMBLD

		self.loss_weights = default_weights.copy()
		if loss_weights is not None:
			for k, v in loss_weights.items():
				self.loss_weights[k] = v

		self.losses_to_plot = []  # Store losses for review later

		if custom_lrs is not None:
			for attr in custom_lrs:
				assert hasattr(SMBLD, attr), f"attr '{attr}' not in SMBLD."

		self.param_group = SMBLDMeshParamGroup(SMBLD, param_group, custom_lrs)

		self.optimizer = torch.optim.Adam(self.param_group, lr=lr)
		self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: lr * (
			lr_decay) ** epoch)  # Decay of learning rate

		with torch.no_grad():
			self.prev_mesh = self.SMBLD.get_meshes()
			self.prev_verts, _ = SMBLD.get_verts()  # Get template verts to use for ARAP method

		self.n_verts = self.prev_verts.shape[1]

		# Sample from target meshes - an equal number to the SMBLD mesh
		self.target_verts = sample_points_from_meshes(self.target_meshes, 3000)

		self.consider_loss = lambda loss_name: self.loss_weights[f"w_{loss_name}"] > 0 # function to check if loss is non-zero

	def loss(self, src_mesh, src_verts):
		loss = 0 

		if self.consider_loss("chamfer"):
			loss_chamfer, _ = chamfer_distance(self.target_verts,
											src_verts)  # We compare the two sets of pointclouds by computing (a) the chamfer loss

			loss += self.loss_weights["w_chamfer"] * loss_chamfer

		if self.consider_loss("edge"):
			loss_edge = mesh_edge_loss(src_mesh)  # and (b) the edge length of the predicted mesh
			loss += self.loss_weights["w_edge"] * loss_edge

		if self.consider_loss("normal"):
			loss_normal = mesh_normal_consistency(src_mesh)  # mesh normal consistency
			loss += self.loss_weights["w_normal"] * loss_normal

		if self.consider_loss("laplacian"):
			loss_laplacian = mesh_laplacian_smoothing(src_mesh, method="uniform")  # mesh laplacian smoothing
			loss += self.loss_weights["w_laplacian"] * loss_normal	

		if self.consider_loss("arap"):
			for n in range(len(self.target_meshes)):
				loss_arap = arap_loss(self.prev_mesh, self.prev_verts, src_verts, mesh_idx=n)
				loss += self.loss_weights["w_arap"] * loss_arap

		return loss, loss_chamfer

	def step(self, epoch):
		"""Runs step of Stage, calculating loss, and running the optimiser"""

		src_verts, src_faces = self.SMBLD.get_verts()
		new_src_mesh = ARAPMeshes(src_verts, src_faces)

		loss, loss_chamfer = self.loss(new_src_mesh, src_verts)
		self.losses_to_plot.append(loss_chamfer)

		# with torch.no_grad():
		# 	## Before stepping, save current verts for next step of ARAP
		# 	self.prev_mesh = new_src_mesh.clone()
		# 	self.prev_verts = src_verts.clone()
	
		# Optimization step
		loss.backward()
		self.optimizer.step()
		self.scheduler.step()  # Update LR


		return loss, loss_chamfer

	def run(self, plot=False):
		"""Run the entire Stage"""

		with tqdm(np.arange(self.n_it)) as tqdm_iterator:
			for i in tqdm_iterator:
				self.optimizer.zero_grad()  # Initialise optimiser
				loss, loss_chamfer = self.step(i)

				tqdm_iterator.set_description(f"STAGE = {self.name}, TOT_LOSS = {loss:.6f}, LOSS_CHAMF = {loss_chamfer:.6f}")  # Print the losses

		if plot:
			figtitle = f"{self.name}, its = {self.n_it}"
			plot_meshes(self.target_meshes, self.SMBLD.get_meshes(), self.mesh_names, title=self.name, figtitle=figtitle,
							 out_dir=os.path.join(self.out_dir, "meshes"))

