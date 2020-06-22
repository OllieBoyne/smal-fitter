"""Using the method described in https://pytorch3d.org/tutorials/deform_source_mesh_to_target_meshes,

Attempts to optimise the SMAL model such that the mesh best fits with the input mesh"""

import sys, os, csv
sys.path.append(os.path.dirname(sys.path[0]))


import torch
from torch import nn
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
from utils import equal_3d_axes
import numpy as np
import tqdm

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl


from smbld_mesh import SMALMeshX

# Set the device
device = torch.device("cuda:0,1")

def load_target_meshes(mesh_dir, sorting = lambda arr: arr, n_meshes=None, frame_step=1):
    """Given a dir of meshes, loads all and returns mesh names, and meshes as Mesh object.
    
    sorting: function for sorting the files in the mesh_dir (by default does nothing)"""
    # load all meshes
    mesh_names = []
    all_verts, all_faces_idx = [], []

    obj_list = sorting(os.listdir(mesh_dir))
    if n_meshes is not None: obj_list = obj_list[:n_meshes]
    obj_list = obj_list[::frame_step]

    for obj_file in obj_list:
        mesh_names.append(obj_file[:-4]) # Get name of mesh
        target_obj = os.path.join(mesh_dir, obj_file)
        verts, faces, aux = load_obj(target_obj, load_textures=False) # Load mesh with no textures
        faces_idx = faces.verts_idx.to(device) # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
        verts = verts.to(device) # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh

        # Center and scale for normalisation purposes
        centre = verts.mean(0)
        verts = verts - centre
        scale = max(verts.abs().max(0)[0])
        verts = verts/scale

        # ROTATE TARGET MESH TO GET IN DESIRED DIRECTION
        R1 = torch.FloatTensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).cuda()# TARGET NEEDS A 90 DEGREE ROTATION ABOUT Y AXIS
        R2 = torch.FloatTensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).cuda() # 90 degree rotation about z
        verts = torch.mm(verts, R1.T)
        verts = torch.mm(verts, R2.T)

        all_verts.append(verts), all_faces_idx.append(faces_idx)

    print(f"{len(all_verts)} target meshes loaded.")

    target_meshes = Meshes(verts=all_verts, faces=all_faces_idx) # All loaded target meshes together
    return mesh_names, target_meshes

class Stage:
    """Defines a stage of optimisation, the optimisation parameters for the stage, ..."""
    def __init__(self, n_it, parameters, SMAL, target_meshes, mesh_names=[], name="optimise",
    loss_weights = dict(w_chamfer = 2.0, w_edge = 1.0,    w_normal = 0.01, w_laplacian = 0.1, w_scale = 0.001),
    lr = 1e-3, momentum = 0.9, lr_decay=1.0,
    out_dir = "fit_to_mesh_outputs"):
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
        self.SMAL = SMAL

        self.loss_weights = loss_weights
        self.losses = [] # Store losses for review later

        self.log = tqdm.tqdm(total=n_it)

        #self.optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum)
        self.optimizer = torch.optim.Adam(parameters, lr = lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: lr * (lr_decay)**epoch) # Decay of learning rate
           
    def loss(self, src_mesh, sample_target, sample_src):
        loss_chamfer, _ = chamfer_distance(sample_target, sample_src)     # We compare the two sets of pointclouds by computing (a) the chamfer loss
        loss_edge = mesh_edge_loss(src_mesh)     # and (b) the edge length of the predicted mesh
        loss_normal = mesh_normal_consistency(src_mesh)     # mesh normal consistency
        loss_laplacian = mesh_laplacian_smoothing(src_mesh, method="uniform")     # mesh laplacian smoothing

        loss = loss_chamfer * self.loss_weights["w_chamfer"] + loss_edge * self.loss_weights["w_edge"] + \
            loss_normal * self.loss_weights["w_normal"] + loss_laplacian * self.loss_weights["w_laplacian"]
            # Weighted sum of the losses

        return loss

    def step(self, epoch):
        """Runs step of Stage, calculating loss, and running the optimiser"""
        
        new_src_mesh = self.SMAL.get_meshes()

        # We sample 5k points from the surface of each mesh    
        sample_target = sample_points_from_meshes(self.target_meshes, 3000)
        sample_src = sample_points_from_meshes(new_src_mesh, 3000)
        
        loss = self.loss(new_src_mesh, sample_target, sample_src)
        self.losses.append(loss)
        self.log.set_description('total_loss = %.6f' % loss) # Print the losses
               
        # Optimization step
        loss.backward()
        self.optimizer.step()
        self.scheduler.step() # Update LR
        self.log.update()

    def run(self):
        """Run the entire Stage"""
        for i in range(self.n_it):
            self.optimizer.zero_grad() # Initialise optimiser
            self.step(i)

        plot_pointclouds(self.target_meshes, self.SMAL.get_meshes(), self.mesh_names, title=self.name, out_dir = self.out_dir)   

    def plot_losses(self, out_src):
        """Plots losses against iterations, saves under out_src"""
        fig, ax = plt.subplots()
        plt.plot(self.losses)
        fig.savefig(out_src)
        plt.close(fig)


def plot_mesh_pointcloud(ax, mesh, label="", colour="blue", 
                            equalize = True, zoom = 1.5):
    """Given a Meshes object, plots the mesh on ax (ax must have projection=3d).
   
    equalize = adjust axis limits such that all axes are equal"""

    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    x, y, z = verts.clone().detach().cpu().unbind(1)
    
    s = ax.scatter3D(x, y, z, c = colour, label=label, alpha=0.3)

    if equalize:
        equal_3d_axes(ax, x, y, z, zoom = zoom)

    return s, (x,y,z)

def plot_pointclouds(target_meshes, src_meshes, mesh_names=[], title="", out_dir = "fit_to_mesh_outputs"):
    """Plot and save fig of point clouds, with 3 figs side by side:
    [target mesh, src_mesh, both]"""

    for n in range(len(target_meshes)):
        fig = plt.figure(figsize=(15, 5))
        axes = [fig.add_subplot(int(f"13{n}"), projection="3d") for n in range(1,4)]
        
        for ax in axes:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

        colours = ["green", "blue"]
        labels = ["target", "SMAL"]
        for i, mesh in enumerate([target_meshes[n], src_meshes[n]]):
            for ax in [axes[1 + i==1], axes[2]]:
                plot_mesh_pointcloud(ax, mesh, colour=colours[i], label=labels[i])

        for ax in axes:

            ax.set_title(title)
            ax.legend()  

        if mesh_names == []: name = n
        else: name = mesh_names[n]

        plt.savefig(f"{out_dir}/{name} - {title}.png") # ADD BETTER NAMING CONVENTION TODO CONSIDER MESH NAMES (PASS THIS TO STAGE OBJECT?)
        plt.close(fig)

def optimise_to_static_meshes():
    """Individually optimise the SMAL model to 13 different target meshes, through a SMAL parameter optimisation stage and a vertex deformation stage. Save outputs as numpy arrays."""

    mesh_names, target_meshes = load_target_meshes(mesh_dir = r"/data/cvfs/ob312/smal_fitter/static_meshes/")

    n_batch = batch_size = len(target_meshes) # Size of all meshes in batch

    # FIRST, OPTIMISE STANDARD SMAL MODEL
    # SMAL Mesh params to optimise
    n_betas = 20

    SMAL = SMALMeshX(n_batch=n_batch)

    # Load and plot initial SMAL Mesh
    verts, faces_idx = SMAL.get_verts()
    src_mesh = Meshes(verts, faces_idx)
    plot_pointclouds(target_meshes, src_mesh, mesh_names = mesh_names, title = "Initialisation")

    # Other parameters to optimise - vertex deform
    deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True) # Tensor to optimise

    Niter_smal = 200 # Number of optimization steps
    Niter_deform = 0 # Number of optimization steps

    stage_smal = Stage(200, SMAL.model_params, SMAL, target_meshes = target_meshes,
                    name = "SMAL fit", mesh_names = mesh_names, 
                    lr = 1e-1, momentum = 0.95)

    stage_smal_2 = Stage(1000, SMAL.model_params, SMAL, target_meshes = target_meshes,
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
            plot_mesh_pointcloud(ax, target_meshes[k])
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
    SMAL = SMALMeshX(n_batch = n_batch)
    n_verts = SMAL.n_verts

    Niter_smal = 1000 # Number of optimization steps

    #  VARY ALL PARAMETERS, INCLUDING DEFORMATIONS
    stage_pos = Stage(Niter_smal, SMAL.model_params, SMAL, target_meshes=target_meshes,
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
    SMAL = SMALMeshX(n_batch = init_meshes)
    n_verts = SMAL.n_verts

    Niter_smal = 750 # Number of optimization steps

    #  VARY ALL PARAMETERS
    stage_shape = Stage(Niter_smal, SMAL.model_params, SMAL, target_meshes=target_meshes[:init_meshes],
            name = "Initial fit",
            lr = 0.1, momentum = 0.95, lr_decay=0.997,
            out_dir = dog_dir)

    stage_shape.run()
    stage_shape.plot_losses(out_src = os.path.join(dog_dir, "losses_shape.png"))

    # Run next stage - fitting pose to all animations
    Niter_pose = 500
    pose_SMAL = SMALMeshX(n_batch = n_batch, fixed_betas=True) # fix betas across frames
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