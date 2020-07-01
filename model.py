"""Produce new SMBLD model based on input verts and SMAL params"""

import numpy as np
from smbld_model.config import SMPL_MODEL_PATH, SMPL_DATA_PATH
import pickle as pkl 
import os
from sklearn.decomposition import PCA

joinp = os.path.join

def produce_new_shapedir(verts, n_betas=20):
    """Given a matrix of batch of vertices, run PCA through SVD in order to identify 
    a certain number of shape parameters to best describe the vert shape.

    :param verts: (N x V x 3) array
    :param n_betas: Number of betas to be fitted to, B
    
    :return vtemplate: (V x 3) Array of new template vertices
    :return shapedir: (nbetas x 3V) Matrix that maps an array of betas to a set of vertex deformations from the template verts
    """

    N, V, _ = verts.shape
    K = min(N, V, n_betas)

    if n_betas > K: 
        print(f"Insufficient size for {n_betas} betas. Using K = {K}")
        n_betas = K

    v_template = verts.mean(axis=0) # set new template verts
    offsets = (verts - v_template).reshape(N, 3*V)

    pca = PCA(n_components = K)

    fit = pca.fit(offsets)
    shapedir = fit.components_.T

    return v_template, shapedir

def get_betas(verts, v_template, shapedir):
    """Given a target sets of verts, template verts, and a shapedir,
    identifies the shape parameters to predict the verts
    
    :param verts: (N x V x 3)
    :param v_template: (V x 3)
    :param shapedir: (nbetas x 3V)
    
    :return betas: (N x nbetas)"""

    N, V, _ = verts.shape
    _, B = shapedir.shape

    offsets = (verts - v_template).reshape(N, 3*V)

    betas = np.zeros((N, B))

    for n in range(N):
        sol = np.linalg.lstsq(shapedir, offsets[n])[0]
        betas[n] = sol

    return betas


def save_new_model(verts, n_betas=20, out_dir = "smbld_model/new_model",
    shape_family=1):
    """Using original SMPL data source, produce new pkl files with:
     - modified shape dir
     - modified mean shapes 
    """

    outfile_model = joinp(out_dir, "model.pkl")
    outfile_data = joinp(out_dir, "data.pkl")

    ### SAVE BASE MODEL
    # -- Load SMPL params --
    with open(SMPL_MODEL_PATH, "rb") as f:
        dd = pkl.load(f, fix_imports = True, encoding="latin1")

    new_vtemplate, new_shapedir = produce_new_shapedir(verts, n_betas=n_betas)
    dd["shapedirs"] = new_shapedir
    dd["v_template"] = new_vtemplate

    with open(outfile_model, "wb") as outfile:
        pkl.dump(dd, outfile)

    ### SAVE MEANS
    with open(SMPL_DATA_PATH, "rb") as f:
        data = pkl.load(f, fix_imports = True, encoding="latin1")

    new_betas = get_betas(verts, new_vtemplate, new_shapedir)
    beta_means = new_betas.mean(axis=0)
    beta_covs = np.cov(new_betas.T)

    # clear cluster means etc, remake
    data['cluster_labels'] = ['unity_dog']
    data['cluster_means'] = [beta_means]
    data['cluster_cov'] = [beta_covs]

    with open(outfile_data, "wb") as outfile:
        pkl.dump(data, outfile)


  
if __name__ == "__main__":

    data = np.load("static_fits_output/smbld_params_arap.npz")
    verts = data["verts"]

    save_new_model(verts)