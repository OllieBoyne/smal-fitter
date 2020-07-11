# smal_fitter

This library of scripts provides Mesh-to-Mesh fitting and model generation of a modified version of the SMAL model, introduced in the paper '3D Menagerie: Modeling the 3D shape and pose of animals' (Zuffi et al, 2016).

The scripts are intended to produce a new version of the model from a set of input `.obj` files, and is capable of

- Optimising the SMBLD model to `.obj` models, and then applying vertex deformations for a smooth fit.
- Fitting this model to animation sequences, provided as `.obj` files.
- Using these fits to produce a new shape space for the models generated.
- Visualising shape spaces and animation sequences.

## Mesh fitting

Mesh fitting allows for the modification of paramaterised meshes, or vertex deformations, to fit an input mesh to the `.obj` files provided.

As Rigid As Possible (ARAP) regularisation can be provided for mesh fitting. This requires the `pytorch-arap` module to also be installed (see setup).

## Setup

- To install, clone the repository with pytorch_arap submodule, using

`
git clone --recurse-submodules https://github.com/OllieBoyne/smal-fitter
`

- **OPTIONAL**: For improved ARAP Speed, install torch_batch_svd, from https://github.com/KinglittleQ/torch-batch-svd. If this is not used, `torch.svd` will be used.

## Dataset

This module was trained using the Unity `Dog Big Pack` https://assetstore.unity.com/packages/3d/characters/dog-big-pack-105660, but will work on any set of .obj files.