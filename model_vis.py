"""Visualise a generated SMAL type model, producing animations of changing key parameters"""

from smbld_model.config import SMPL_DATA_PATH, SMPL_MODEL_PATH
from smbld_model.smbld_mesh import SMBLDMesh
import numpy as np

from matplotlib import pyplot as plt
from utils import plot_mesh, save_animation

models = {
    "default": dict(data_path = SMPL_DATA_PATH, model_path = SMPL_MODEL_PATH),
    "new": dict()
}

def vis_shape_params(data_path, model_path, fps = 15):
    """Load SMBLD model. Wiggle each shape param in turn"""

    mesh = SMBLDMesh(model_path=model_path, data_path = data_path)

    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    trisurfs = []

    num_steps = 10
    num_betas = 20

    shape_range = np.linspace(-1, 1, num_steps)
    n_frames = num_steps * num_betas

    def anim(i):
        [t.remove() for t in trisurfs]  # clear existing meshes
        
        ### update SMBLD
        cur_beta_idx = i // num_steps
        cur_step = i % num_steps
        val = shape_range[cur_step]

        mesh.multi_betas[0, cur_beta_idx] = val  # Update betas
        plot_mesh(ax, mesh.get_meshes()) # get mesh

        ### update text
        fig.suptitle(f"S{cur_beta_idx} : {val:.3f}")

    save_animation(fig, anim, n_frames=n_frames, fmt="gif", title="model_shape_param", fps = fps)

if __name__ == "__main__":

    vis_shape_params(**models["default"], fps = 2)