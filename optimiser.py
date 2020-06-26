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

from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from utils import plot_pointclouds, plot_meshes
import numpy as np
import os
from smbld_model.smbld_mesh import SMBLDMesh
from time import perf_counter

# default_weights = dict(w_chamfer=2.0, w_edge=1.0, w_normal=0.01, w_laplacian=0.1, w_scale=0.001, w_arap=0.001)
default_weights = dict(w_chamfer=2.0, w_edge=0, w_normal=0, w_laplacian=0, w_arap=0)

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
			ax.plot(np.arange(it_start, it_start+n_it), stage.losses, label=stage.name)
			it_start += n_it

		ax.legend()
		out_src = os.path.join(self.out_dir, out_src+".png")
		fig.savefig(out_src)
		plt.close(fig)

	def add_stage(self, stage):
		self.stages.append(stage)


class Stage:
	"""Defines a stage of optimisation, the optimisation parameters for the stage, ..."""

	def __init__(self, n_it: int, parameters: list, SMBLD: SMBLDMesh, target_meshes: Meshes, mesh_names=[], name="optimise",
				 loss_weights=None, lr=1e-3, lr_decay=1.0,out_dir="static_fits_output"):
		"""
		n_its = integer, number of iterations in stage
		parameters = list of items over which to be optimised
		get_mesh = function that returns Mesh object for identifying losses
		name = name of stage

		other_loss_functions = Any other functions describing loss, unique to this stage

		lr_decay = factor by which lr decreases at each epoch"""
		self.n_it = n_it
		self.parameters = parameters
		self.name = name
		self.out_dir = out_dir
		self.target_meshes = target_meshes
		self.mesh_names = mesh_names
		self.SMBLD = SMBLD

		self.loss_weights = default_weights.copy()
		if loss_weights is not None:
			for k, v in loss_weights.items():
				self.loss_weights[k] = v

		self.losses = []  # Store losses for review later

		# self.optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum)
		self.optimizer = torch.optim.Adam(parameters, lr=lr)
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

		## if verts are undeformed from last iteration, no ARAP loss
		# if ((self.prev_verts - src_verts) ** 2).mean() == 0 or self.loss_weights["w_arap"] == 0:
		# 	loss_arap = 0   # no energy if vertices are not deformed
		# else:
			

		return loss

	def step(self, epoch):
		"""Runs step of Stage, calculating loss, and running the optimiser"""

		src_verts, src_faces = self.SMBLD.get_verts()
		new_src_mesh = ARAPMeshes(src_verts, src_faces)

		loss = self.loss(new_src_mesh, src_verts)
		self.losses.append(loss)

		with torch.no_grad():
			## Before stepping, save current verts for next step of ARAP
			self.prev_mesh = new_src_mesh.clone()
			self.prev_verts = src_verts.clone()

		# Optimization step
		loss.backward()

		self.optimizer.step()
		self.scheduler.step()  # Update LR


		return loss

	def run(self, plot=False):
		"""Run the entire Stage"""

		with tqdm(np.arange(self.n_it)) as tqdm_iterator:
			for i in tqdm_iterator:
				self.optimizer.zero_grad()  # Initialise optimiser
				loss = self.step(i)

				tqdm_iterator.set_description(f"STAGE = {self.name}, LOSS = {loss:.6f}")  # Print the losses

		if plot:
			figtitle = f"{self.name}, its = {self.n_it}"
			plot_meshes(self.target_meshes, self.SMBLD.get_meshes(), self.mesh_names, title=self.name, figtitle=figtitle,
							 out_dir=os.path.join(self.out_dir, "pointclouds"))

