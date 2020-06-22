"""Using the method described in https://pytorch3d.org/tutorials/deform_source_mesh_to_target_meshes,

Attempts to optimise the SMAL model such that the mesh best fits with the input mesh"""

import sys, os, csv
sys.path.append(os.path.dirname(sys.path[0]))
from utils import plot_pointcloud, plot_pointclouds
import torch
from torch import nn
from mesh_loader import load_target_meshes
from pytorch3d.structures import Meshes
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from smbld_mesh import SMBLDMesh
from optimiser import Stage

# Set the device
device = torch.device("cuda:0,1")



def optimise_to_static_meshes():
    """Individually optimise the SMAL model to 13 different target meshes, through a SMAL parameter optimisation stage and a vertex deformation stage. Save outputs as numpy arrays."""

    mesh_names, target_meshes = load_target_meshes(mesh_dir = r"/data/cvfs/ob312/smal_fitter/static_meshes/")

    n_batch = batch_size = len(target_meshes) # Size of all meshes in batch

    # FIRST, OPTIMISE STANDARD SMAL MODEL
    # SMAL Mesh params to optimise
    n_betas = 20

    SMAL = SMBLDMesh(n_batch=n_batch)

    # Load and plot initial SMAL Mesh
    verts, faces_idx = SMAL.get_verts()
    src_mesh = Meshes(verts, faces_idx)
    plot_pointclouds(target_meshes, src_mesh, mesh_names = mesh_names, title = "Initialisation")

    # Other parameters to optimise - vertex deform
    deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True) # Tensor to optimise

    Niter_smal = 200 # Number of optimization steps
    Niter_deform = 0 # Number of optimization steps

    stage_smal = Stage(200, SMAL.smbld_params, SMAL, target_meshes = target_meshes,
                       name = "SMAL fit", mesh_names = mesh_names,
                       lr = 1e-1, momentum = 0.95)

    stage_smal_2 = Stage(1000, SMAL.smbld_params, SMAL, target_meshes = target_meshes,
                         name = "SMAL fit 2", mesh_names = mesh_names,
                         lr = 5e-2, momentum = 0.95)

    # stage_deform = Stage(Niter_deform, SMAL.deform_params, SMAL, target_meshes = target_meshes,
    #                 name = "Deform fit", mesh_names = mesh_names, 
    #                 lr = 1.0, momentum = 0.9)                

    stages = [stage_smal]#, stage_smal_2]

    for stage in stages:
        stage.run()
        stage.plot_losses(f"/data/cvfs/ob312/smal_fitter/fit_to_mesh_outputs/losses-{stage.name}.png")

    # EXPORT OUTPUT
    out_dir = r"/data/cvfs/ob312/smal_fitter/fit_to_mesh_outputs/data"
    with open(os.path.join(out_dir, "mesh_names.csv"), "w", newline="") as outfile:
        w = csv.writer(outfile)
        verts_per_mesh = src_mesh.num_verts_per_mesh().cpu().detach().numpy()
        w.writerows([[mesh_names[n], verts_per_mesh[n]] for n in range(batch_size)])

    # # Save all SMAL params
    for param in ["global_rot", "joint_rot", "multi_betas", "trans"]:
        arr = getattr(SMAL, param).cpu().detach().numpy()
        np.save(os.path.join(out_dir, param), arr)
        
    # Save Mesh deformations - shape is (total_verts, 3)
    #np.save(os.path.join(out_dir, "deform_verts"), deform_verts.cpu().detach().numpy())

def vis_tpose():
    """Visualise all T-pose for all dogs"""
    src = r"/data/cvfs/ob312/smal_fitter/static_meshes"
    _, target_meshes = load_target_meshes(mesh_dir = src)

    dpi = 300
    fig = plt.figure(figsize = (3000/dpi, 3000/dpi))
    N_dogs = len(target_meshes)
    n_rows = n_cols = int(N_dogs ** 0.5) + 1

    k = 0
    for i in range(n_rows):
        for j in range(n_rows):
            ax = fig.add_subplot(n_rows, n_cols, k+1, projection="3d")
            plot_pointcloud(ax, target_meshes[k])
            k+=1
            if k == N_dogs: break # Once plotted all dogs
    
    plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
    plt.tight_layout()
    plt.savefig(r"/data/cvfs/ob312/smal_fitter/misc_figs/t-pose.png", dpi = dpi)


def fit_to_animations(dog = "amstaff"):
    """For each animation sequence, fits SMAL parameters + vertex deformations, with the vertex_deformations"""

    animated_output_dir = f"/data/cvfs/ob312/smal_fitter/animation_fit_to_mesh_outputs"
    # Make folders
    dog_dir = os.path.join(animated_output_dir, dog)
    dog_data_dir = os.path.join(dog_dir, "data")
    if not os.path.isdir(dog_dir):
        os.mkdir(dog_dir), os.mkdir(dog_data_dir)

    # Load Amstaff only atm
    _, target_meshes = load_target_meshes(mesh_dir = f"/data/cvfs/ob312/smal_fitter/animated_meshes/{dog}",
        sorting = lambda arr: sorted(arr, key = lambda i: int(i.split("_")[-1].split(".")[0])),
        n_meshes = 5)

    n_batch = batch_size = len(target_meshes) # Size of all meshes in batch

    # FIRST, OPTIMISE STANDARD SMAL MODEL
    # SMAL Mesh params to optimise
    n_betas = 40

    # SMAL = SMALMesh(n_betas=n_betas, n_batch=n_batch, fix_deform_across_meshes=True)
    SMAL = SMBLDMesh(n_batch = n_batch)
    n_verts = SMAL.n_verts

    Niter_smal = 1000 # Number of optimization steps

    #  VARY ALL PARAMETERS, INCLUDING DEFORMATIONS
    stage_pos = Stage(Niter_smal, SMAL.smbld_params, SMAL, target_meshes=target_meshes,
                      name = "Animation fit",
                      lr = 0.8, momentum = 0.95, lr_decay=0.999,
                      out_dir = dog_dir)

    stage_pos.run()
    stage_pos.plot_losses(out_src = os.path.join(dog_dir, "losses.png"))

    SMAL.numpy_save(dog_data_dir)

def fit_to_animation_2stage(dog = "amstaff"):
    """For animation sequence of dog, fits all SMAL params to first few frames, then pose params to all frames"""

    animated_output_dir = f"/data/cvfs/ob312/smal_fitter/animation_fit_to_mesh_outputs"
    # Make folders
    dog_dir = os.path.join(animated_output_dir, dog)
    dog_data_dir = os.path.join(dog_dir, "data")
    if not os.path.isdir(dog_dir):
        os.mkdir(dog_dir), os.mkdir(dog_data_dir)

    # Load target meshes only atm
    _, target_meshes = load_target_meshes(mesh_dir = f"/data/cvfs/ob312/smal_fitter/animated_meshes/{dog}",
        sorting = lambda arr: sorted(arr, key = lambda i: int(i.split("_")[-1].split(".")[0])),
        frame_step=4)

    n_batch = batch_size = len(target_meshes) # Size of all meshes in batch

    # FIRST, OPTIMISE STANDARD SMAL MODEL
    # SMAL Mesh params to optimise
    n_betas = 40

    # SMAL = SMALMesh(n_betas=n_betas, n_batch=n_batch, fix_deform_across_meshes=True)
    init_meshes = 1 # first select 5 meshes for model fitting
    SMAL = SMBLDMesh(n_batch = init_meshes)
    n_verts = SMAL.n_verts

    Niter_smal = 750 # Number of optimization steps

    #  VARY ALL PARAMETERS
    stage_shape = Stage(Niter_smal, SMAL.smbld_params, SMAL, target_meshes=target_meshes[:init_meshes],
                        name = "Initial fit",
                        lr = 0.1, momentum = 0.95, lr_decay=0.997,
                        out_dir = dog_dir)

    stage_shape.run()
    stage_shape.plot_losses(out_src = os.path.join(dog_dir, "losses_shape.png"))

    # Run next stage - fitting pose to all animations
    Niter_pose = 500
    pose_SMAL = SMBLDMesh(n_batch = n_batch, fixed_betas=True) # fix betas across frames
    # Assign learned parameters to new model
    pose_SMAL.multi_betas = nn.Parameter(SMAL.multi_betas.repeat(n_batch, 1))

    stage_pose = Stage(Niter_pose, [pose_SMAL.global_rot, pose_SMAL.joint_rot, pose_SMAL.trans], pose_SMAL, target_meshes=target_meshes,
                name = "Animation fit",
                lr = 0.1, momentum = 0.8, lr_decay=0.995,
                out_dir = dog_dir)

    stage_pose.run()
    stage_pose.plot_losses(out_src = os.path.join(dog_dir, "losses_pose.png"))

    pose_SMAL.numpy_save(dog_data_dir)

# optimise_to_static_meshes()
fit_to_animation_2stage("amstaff_new_prior")
# [fit_to_animations(d) for d in os.listdir(r"/data/cvfs/ob312/smal_fitter/animated_meshes/")]
# vis_tpose()