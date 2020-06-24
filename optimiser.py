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

from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from utils import plot_pointclouds
import numpy as np
import os
from smbld_model.smbld_mesh import SMBLDMesh

default_weights = dict(w_chamfer=2.0, w_edge=1.0, w_normal=0.01, w_laplacian=0.1, w_scale=0.001, w_arap=0.01)

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

		self.loss_weights = default_weights
		if loss_weights is not None:
			for k, v in loss_weights.items():
				self.loss_weights[k] = v

		self.losses = []  # Store losses for review later

		# self.optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum)
		self.optimizer = torch.optim.Adam(parameters, lr=lr)
		self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: lr * (
			lr_decay) ** epoch)  # Decay of learning rate

		with torch.no_grad():
			self.prev_mesh = self.SMBLD.get_meshes(arap=True)
			self.prev_verts, _ = SMBLD.get_verts()  # Get template verts to use for ARAP method


	def loss(self, src_mesh, sample_target, sample_src):
		loss_chamfer, _ = chamfer_distance(sample_target,
										   sample_src)  # We compare the two sets of pointclouds by computing (a) the chamfer loss
		loss_edge = mesh_edge_loss(src_mesh)  # and (b) the edge length of the predicted mesh
		loss_normal = mesh_normal_consistency(src_mesh)  # mesh normal consistency
		loss_laplacian = mesh_laplacian_smoothing(src_mesh, method="uniform")  # mesh laplacian smoothing

		## if verts are undeformed from last iteration, no ARAP loss
		verts_deformed = src_mesh.verts_padded()
		if ((self.prev_verts - verts_deformed) ** 2).mean() == 0 or self.name=="1 - Initial fit":
			loss_arap = 0   # no energy if vertices are not deformed
		else:
			loss_arap = arap_loss(self.prev_mesh, self.prev_verts, src_mesh.verts_padded())
			print("USING ARAP", loss_chamfer, 0.001*loss_arap)

		# Weighted sum of the losses
		# loss = loss_chamfer * self.loss_weights["w_chamfer"] + loss_edge * self.loss_weights["w_edge"] + \
		# 	   loss_normal * self.loss_weights["w_normal"] + loss_laplacian * self.loss_weights["w_laplacian"] + \
		# 	loss_arap * self.loss_weights["w_arap"]

		loss = loss_chamfer + 0.001 * loss_arap

		return loss

	def step(self, epoch):
		"""Runs step of Stage, calculating loss, and running the optimiser"""

		new_src_mesh = self.SMBLD.get_meshes(arap=True)

		# We sample 5k points from the surface of each mesh
		sample_target = sample_points_from_meshes(self.target_meshes, 3000)
		sample_src = sample_points_from_meshes(new_src_mesh, 3000)

		loss = self.loss(new_src_mesh, sample_target, sample_src)
		self.losses.append(loss)

		with torch.no_grad():
			## Before stepping, save current verts for next step of ARAP
			self.prev_mesh = self.SMBLD.get_meshes(arap=True)
			self.prev_verts, _ = self.SMBLD.get_verts()

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
			plot_pointclouds(self.target_meshes, self.SMBLD.get_meshes(), self.mesh_names, title=self.name,
							 out_dir=os.path.join(self.out_dir, "pointclouds"))

	def plot_losses(self, out_src):
		"""Plots losses against iterations, saves under out_src"""
		fig, ax = plt.subplots()
		plt.plot(self.losses)
		fig.savefig(out_src)
		plt.close(fig)