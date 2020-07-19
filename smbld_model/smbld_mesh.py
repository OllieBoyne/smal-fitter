"""Script defining SMALMesh, an object capable of rendering a mesh version of the SMAL Model, for optimising the fit to other, existing meshes.

With modifications now to work with:
- newest SMAL Model
- Newly define scale factor parameters"""

from absl import flags
from pytorch3d.structures import Meshes
import sys, os

sys.path.append(os.path.dirname(sys.path[0]))
from smbld_model.smal_model.smal_torch import SMAL
import torch
from smbld_model.smal_model.smal_torch import batch_rodrigues
import numpy as np
import pickle
from utils import stack_as_batch, try_mkdir
from pytorch_arap.pytorch_arap.arap import ARAPMeshes
from smbld_model.config import SMPL_MODEL_PATH, SMPL_DATA_PATH

nn = torch.nn
opts = flags.FLAGS

kappa_map = {
    "front_left_leg": 7,
    "front_right_leg" : 11,
    "rear_left_leg": 17,
    "rear_right_leg": 21,
    "tail": 25,
    "core": 1, # NOTE: this is linked to head/front legs, will have to reduce them by an equal amount
    "neck": 15, # Head is connected to this
    "head": 16,
    "left_ear": 33,
    "right_ear":34,
}

def batch_global_rigid_transformation(Rs, Js, parent, rotate_base = False, betas_extra=None, device="cuda"):
    """
    Computes absolute joint locations given pose.
    rotate_base: if True, rotates the global rotation by 90 deg in x axis.
    if False, this is the original SMPL coordinate.
    Args:
      Rs: N x 24 x 3 x 3 rotation vector of K joints
      Js: N x 24 x 3, joint locations before posing
      parent: 24 holding the parent id for each index
    Returns
      new_J : `Tensor`: N x 24 x 3 location of absolute joints
      A     : `Tensor`: N x 24 4 x 4 relative joint transformations for LBS.
    """

    # Now Js is N x 24 x 3 x 1
    Js = Js.unsqueeze(-1)
    N = Rs.shape[0]

    if rotate_base:
        print('Flipping the SMPL coordinate frame!!!!')
        rot_x = torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        rot_x = torch.reshape(torch.repeat(rot_x, [N, 1]), [N, 3, 3]) # In tf it was tile
        root_rotation = torch.matmul(Rs[:, 0, :, :], rot_x)
    else:
        root_rotation = Rs[:, 0, :, :]

    Js_orig = Js.clone()

    scaling_factors = torch.ones(N, parent.shape[0], 3).to(device)
    if betas_extra is not None:
        scaling_factors = betas_extra.reshape(-1, 35, 3)
        # debug_only
        # scaling_factors[:, 25:32, 0] = 0.2
        # scaling_factors[:, 7, 2] = 2.0

    scale_factors_3x3 = torch.diag_embed(scaling_factors, dim1=-2, dim2=-1)

    def make_A(R, t):
        # Rs is N x 3 x 3, ts is N x 3 x 1
        R_homo = torch.nn.functional.pad(R, (0,0,0,1,0,0))
        t_homo = torch.cat([t, torch.ones([N, 1, 1]).to(device)], 1)
        return torch.cat([R_homo, t_homo], 2)
    
    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]
    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]

        s_par_inv = torch.inverse(scale_factors_3x3[:, parent[i]])
        rot = Rs[:, i]
        s = scale_factors_3x3[:, i]
        
        rot_new = s_par_inv @ rot @ s

        A_here = make_A(rot_new, j_here)
        res_here = torch.matmul(
            results[parent[i]], A_here)
        
        results.append(res_here)

    # 10 x 24 x 4 x 4
    results = torch.stack(results, dim=1)

    # scale updates
    new_J = results[:, :, :3, 3]

    # --- Compute relative A: Skinning is based on
    # how much the bone moved (not the final location of the bone)
    # but (final_bone - init_bone)
    # ---
    Js_w0 = torch.cat([Js_orig, torch.zeros([N, 35, 1, 1]).to(device)], 2)
    init_bone = torch.matmul(results, Js_w0)
    # Append empty 4 x 3:
    init_bone = torch.nn.functional.pad(init_bone, (3,0,0,0,0,0,0,0))
    A = results - init_bone

    return new_J, A

class SMBLDMesh(SMAL, nn.Module):
    """SMAL Model, with addition of scale factors to individual body parts"""

    def __init__(self, n_batch = 1, fixed_betas = False, device="cuda", shape_family_id = 1,
    model_path = SMPL_MODEL_PATH, data_path = SMPL_DATA_PATH, num_betas=20, **kwargs):


        SMAL.__init__(self, model_path=model_path, data_path=data_path, opts = opts, shape_family_id=shape_family_id,
                      align = False)
        nn.Module.__init__(self)
    
        self.use_smal_betas = True
        self.n_batch = n_batch
        self.device = device

        self.v_template=  self.v_template.to(device)
        self.faces = self.f
        faces_single = torch.from_numpy(self.faces.astype(np.float32)).to(device)
        self.faces_batch = stack_as_batch(faces_single, n_batch)

        self.n_verts = self.v_template.shape[0]

        #parameters
        self.global_rot = nn.Parameter(torch.full((n_batch, 3), 0.0, device = device, requires_grad=True))
        self.joint_rot = nn.Parameter(torch.full((n_batch, 34, 3), 0.0, device = device, requires_grad=True))
        self.trans = nn.Parameter(torch.full((n_batch, 3,), 0.0, device = device, requires_grad=True))


        self.scale_factors = torch.nn.Parameter(torch.ones((self.parents.shape[0])),
                                                requires_grad = True)

        # This sets up a new set of betas that define the scale factor parameters
        self.num_beta_shape = self.n_betas = 20
        self.num_betascale = 7
        leg_joints = list(range(7,11)) + list(range(11,15)) + list(range(17,21)) + list(range(21,25))
        tail_joints = list(range(25, 32))
        ear_joints = [33, 34]

        beta_scale_mask = torch.zeros(35, 3, 7).to(device)
        beta_scale_mask[leg_joints, [2], [0]] = 1.0 # Leg lengthening
        beta_scale_mask[leg_joints, [0], [1]] = 1.0 # Leg fatness
        beta_scale_mask[leg_joints, [1], [1]] = 1.0 # Leg fatness
        
        beta_scale_mask[tail_joints, [0], [2]] = 1.0 # Tail lengthening
        beta_scale_mask[tail_joints, [1], [3]] = 1.0 # Tail fatness
        beta_scale_mask[tail_joints, [2], [3]] = 1.0 # Tail fatness
        
        beta_scale_mask[ear_joints, [1], [4]] = 1.0 # Ear y
        beta_scale_mask[ear_joints, [2], [5]] = 1.0 # Ear z

        self.beta_scale_mask = torch.transpose(beta_scale_mask.reshape(35*3, self.num_betascale), 0, 1)

        self.fixed_betas = fixed_betas

        self.num_betas = num_betas # number of used betas

        max_betas = self.shapedirs.shape[0]
        assert max_betas >= self.num_betas, f"Insufficient number of betas in shapedir (Requested {self.num_betas}, shapedir has {max_betas})"

        # Load mean betas from SMAL model
        with open(data_path, "rb") as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            smal_data = u.load()
            shape_family = self.shape_family_id # Canine is family=1
            if shape_family == -1:
                self.mean_betas = torch.zeros((41)).to(device)
            else:
                loaded_betas = smal_data['cluster_means'][shape_family]
                if len(loaded_betas) < max_betas:
                    loaded_betas = np.pad(loaded_betas, (0, self.num_betas-len(loaded_betas))) # pad with 0s to max shape
                self.mean_betas = torch.FloatTensor(loaded_betas).to(device)

        multi_betas = self.mean_betas[:self.num_betas]
        multi_betas_scale = torch.zeros(self.num_betascale).float().to(device)

        multi_betas = torch.cat([multi_betas, multi_betas_scale], dim = 0)

        if self.fixed_betas:
            self.multi_betas = nn.Parameter(multi_betas.repeat(1, 1))
        else:
            self.multi_betas = nn.Parameter(multi_betas.repeat(self.n_batch, 1))

        self.deform_verts = nn.Parameter(torch.zeros((n_batch, self.n_verts, 3), device=device, requires_grad=True))

        self.smbld_shape = [self.global_rot, self.trans, self.multi_betas]
        self.smbld_params = [self.global_rot, self.joint_rot, self.trans, self.multi_betas] # params of SMBDL model
        self.deform_params = [self.deform_verts]

        self.meshes = self.get_meshes()

    def get_verts(self, return_joints=False):
        """Returns vertices and faces of SMAL Model"""
        # For reference on running the forward() method of SMAL model, see smal3d_renderer.py
        smal_params = self.parameters()

        # Split betas by standard betas, and scale factor betas
        all_betas = self.multi_betas
        betas_pred = all_betas[:, :self.num_betas] # usual betas
        betas_logscale = all_betas[:, self.num_betas:] # Scale factor betas
        betas_scale_pred = torch.exp(betas_logscale @ self.beta_scale_mask) # Scale SF betas correctly

        #betas = betas_pred.repeat(self.n_batch, 1) # Stack Betas correctly if fixed across batch
        #sf = self.scale_factors.repeat(self.n_batch, 1) # Stack Betas correctly if fixed across batch

        verts, joints_3d, R = self(betas_pred,
                torch.cat((self.global_rot, self.joint_rot.view(self.n_batch, -1)), dim = 1),
                betas_scale_pred.to(self.device), trans=self.trans, deform_verts=self.deform_verts)

        if return_joints:
            return verts, self.faces_batch, joints_3d
        return verts, self.faces_batch # each of these have shape (n_batch, n_vert/faces, 3)

    def get_meshes(self):
        """Returns Meshes object of all SMAL meshes."""

        self.meshes = ARAPMeshes(*self.get_verts(), device=self.device)

        return self.meshes

    def __call__(self, beta, theta, betas_extra, deform_verts=None, trans=None, get_skin=True):

        if self.use_smal_betas: # Always use smal betas
            nBetas = beta.shape[1]
        else:
            nBetas = 0
        # 1. Add shape blend shapes
        if nBetas > 0:
            if deform_verts is None:
                v_shaped = self.v_template.to(self.device) + torch.reshape(torch.matmul(beta.cpu(), self.shapedirs[:nBetas, :]),
                                                           [-1, self.size[0], self.size[1]]).to(self.device)
            else:
                v_shaped = self.v_template + deform_verts + torch.reshape(torch.matmul(beta, self.shapedirs[:nBetas, :].to(self.device)),
                                                                   [-1, self.size[0], self.size[1]]).to(self.device)
        else:
            if deform_verts is None:
                v_shaped = self.v_template.unsqueeze(0)
            else:
                v_shaped = self.v_template + deform_verts

        # 2. Infer shape-dependent joint locations.
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor.to(self.device))
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor.to(self.device))
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor.to(self.device))
        J = torch.stack([Jx, Jy, Jz], dim=2)
        # 3. Add pose blend shapes
        # N x 24 x 3 x 3
        Rs = torch.reshape(batch_rodrigues(torch.reshape(theta, [self.n_batch * 35, 3]).cpu()), [-1, 35, 3, 3]).to(self.device)

        # Ignore global rotation.
        pose_feature = torch.reshape(Rs[:, 1:, :, :].to(self.device) - torch.eye(3).to(self.device), [-1, 306]) # torch.eye(3).cuda(device=self.opts.gpu_id)
        v_posed = torch.reshape(
            torch.matmul(pose_feature, self.posedirs.to(self.device)),
            [-1, self.size[0], self.size[1]]) + v_shaped.to(self.device)

        # 4. Get the global joint location
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents,
                                                                  betas_extra=betas_extra, device=self.device)
        # 5. Do skinning:
        num_batch = theta.shape[0]
        weights_t = self.weights.repeat([num_batch, 1]).to(self.device)
        W = torch.reshape(weights_t, [num_batch, -1, 35])
        T = torch.reshape(
            torch.matmul(W, torch.reshape(A, [num_batch, 35, 16])),
            [num_batch, -1, 4, 4])
        v_posed_homo = torch.cat(
            [v_posed, torch.ones([num_batch, v_posed.shape[1], 1]).to(self.device)], 2) #.cuda(device=self.opts.gpu_id)
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))
        verts = v_homo[:, :, :3, 0]
        if trans is None:
            trans = torch.zeros((num_batch, 3)).to(self.device)#.cuda(device=self.opts.gpu_id)
        verts = verts + trans[:, None, :]

        # Get joints:
        self.J_regressor = self.J_regressor.to(self.device)
        joint_x = torch.matmul(verts[:, :, 0], self.J_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.J_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.J_regressor)
        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)

        if get_skin:
            return verts, joints, Rs
        else:
            return joints

    def save_npz(self, out_dir, title="", labels=None):
        """Given a directory, saves a .npz file of all params
        
        labels: optional list of size n_batch, to save as labels for all entries"""

        out = {}
        for param in ["global_rot", "joint_rot", "multi_betas", "trans", "deform_verts"]:
            out[param] = getattr(self, param).cpu().detach().numpy()

        v, f = self.get_verts()
        out["verts"] = v.cpu().detach().numpy()
        out["faces"] = f.cpu().detach().numpy()
        out["labels"] = labels

        out_title = "smbld_params.npz"
        if title != "":
            out_title = out_title.replace(".npz", f"_{title}.npz")

        try_mkdir(out_dir)
        np.savez(os.path.join(out_dir, out_title), **out)

    def load_from_npz(self, loc):
        """Given the location of a .npz file, load previous model"""

        data = np.load(loc)

        for param in ["global_rot", "joint_rot", "multi_betas", "trans"]:
            tensor = torch.from_numpy(data[param]).to(self.device)
            getattr(self, param).data = tensor