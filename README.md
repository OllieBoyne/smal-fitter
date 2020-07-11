# smal_fitter
Fitting Skinned Multi Linear Animal Model (SMAL) to other meshes

## Installation

1. Clone the repository with pytorch_arap submodule, using

`
git clone --recurse-submodules https://github.com/OllieBoyne/smal-fitter
`

OPTIONAL: For improved ARAP Speed, install torch_batch_svd, from https://github.com/KinglittleQ/torch-batch-svd/blob/master/torch_batch_svd/include/utils.h
If this is not used, torch.svd will be used.

## Dataset

This module was trained using the Unity `Dog Big Pack` https://assetstore.unity.com/packages/3d/characters/dog-big-pack-105660, but will work on any set of .obj files.
