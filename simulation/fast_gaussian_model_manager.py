'''
Modified from https://github.com/yzslab/gaussian-splatting-lightning
Modified from https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
'''
import torch
import numpy as np
import time
import warp as wp
from typing import Union
from dataclasses import dataclass
from plyfile import PlyData, PlyElement
import os
import sys
import torch.nn.functional as F
try:
    from e3nn import o3
    import einops
    from einops import einsum
except:
    print("Please run `pip install e3nn einops` to enable SHs rotation")
    sys.exit(0)

wp.init()

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def transform_shs(features, rotation_matrix):
    """
    https://github.com/graphdeco-inria/gaussian-splatting/issues/176#issuecomment-2147223570
    """
    
    if features.shape[1] == 1:
        return features

    features = features.clone()

    shs_feat = features[:, 1:, :]

    ## rotate shs
    P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=shs_feat.dtype, device=shs_feat.device)  # switch axes: yzx -> xyz
    inversed_P = torch.tensor([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ], dtype=shs_feat.dtype, device=shs_feat.device)
    permuted_rotation_matrix = inversed_P @ rotation_matrix @ P
    rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix.cpu())

    # Construction coefficient
    D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)
    D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)
    D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)

    # rotation of the shs features
    one_degree_shs = shs_feat[:, 0:3]
    one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    one_degree_shs = einsum(
        D_1,
        one_degree_shs,
        "... i j, ... j -> ... i",
    )
    one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 0:3] = one_degree_shs

    if shs_feat.shape[1] >= 4:
        two_degree_shs = shs_feat[:, 3:8]
        two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
        two_degree_shs = einsum(
            D_2,
            two_degree_shs,
            "... i j, ... j -> ... i",
        )
        two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
        shs_feat[:, 3:8] = two_degree_shs

        if shs_feat.shape[1] >= 9:
            three_degree_shs = shs_feat[:, 8:15]
            three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
            three_degree_shs = einsum(
                D_3,
                three_degree_shs,
                "... i j, ... j -> ... i",
            )
            three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
            shs_feat[:, 8:15] = three_degree_shs

    return features

def quat_multiply(quaternion0, quaternion1):
    w0, x0, y0, z0 = torch.split(quaternion0, 1, dim=-1)
    w1, x1, y1, z1 = torch.split(quaternion1, 1, dim=-1)
    return torch.concatenate((
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
    ), dim=-1)

def wigner_D(l, alpha, beta, gamma, device):
    r"""Wigner D matrix representation of :math:`SO(3)`.

    It satisfies the following properties:

    * :math:`D(\text{identity rotation}) = \text{identity matrix}`
    * :math:`D(R_1 \circ R_2) = D(R_1) \circ D(R_2)`
    * :math:`D(R^{-1}) = D(R)^{-1} = D(R)^T`
    * :math:`D(\text{rotation around Y axis})` has some property that allows us to use FFT in `ToS2Grid`

    Parameters
    ----------
    l : int
        :math:`l`

    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\alpha` around Y axis, applied third.

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\beta` around X axis, applied second.

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\gamma` around Y axis, applied first.

    Returns
    -------
    `torch.Tensor`
        tensor :math:`D^l(\alpha, \beta, \gamma)` of shape :math:`(2l+1, 2l+1)`
    """
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    alpha = alpha[..., None, None] % (2 * torch.pi)
    beta = beta[..., None, None] % (2 * torch.pi)
    gamma = gamma[..., None, None] % (2 * torch.pi)
    X = o3.so3_generators(l).to(device)
    return torch.matrix_exp(alpha * X[1]) @ torch.matrix_exp(beta * X[0]) @ torch.matrix_exp(gamma * X[1])

def load_array_from_plyelement(plyelement, name_prefix: str):
    names = [p.name for p in plyelement.properties if p.name.startswith(name_prefix)]
    if len(names) == 0:
        print(f"WARNING: '{name_prefix}' not found in ply, create an empty one")
        return np.empty((plyelement["x"].shape[0], 0))
    names = sorted(names, key=lambda x: int(x.split('_')[-1]))
    v_list = []
    for idx, attr_name in enumerate(names):
        v_list.append(np.asarray(plyelement[attr_name]))

    return np.stack(v_list, axis=1)

class GaussianModel(torch.nn.Module):
    def __init__(
            self,
            xyz: np.ndarray,
            scaling: np.ndarray,
            rotation: np.ndarray,
            opacity: np.ndarray,
            features: np.ndarray,
            sh_degrees: int,
            device: torch.device = torch.device("cuda:0"),
            dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self._xyz = torch.tensor(xyz).to(dtype).to(device)
        self._scaling = torch.exp(torch.tensor(scaling).to(dtype).to(device))
        self._rotation = torch.nn.functional.normalize(torch.tensor(rotation).to(dtype).to(device))
        self._opacity = torch.sigmoid(torch.tensor(opacity).to(dtype).to(device))
        self._features = torch.tensor(features).to(dtype).to(device)
        self.sh_degrees = sh_degrees
        
    @property
    def get_scaling(self):
        return self._scaling

    @property
    def get_rotation(self):
        return self._rotation

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return self._features

    @property
    def get_opacity(self):
        return self._opacity

    def scale(self, factor: torch.Tensor):
        self._xyz *= factor.unsqueeze(0)
        self._scaling *= factor.unsqueeze(0)
        
    def translate(self, translation: torch.Tensor):
        self._xyz += translation.unsqueeze(0)
    
    def rotate(self, quaternions: torch.tensor):
        # convert quaternions to rotation matrix
        rotation_matrix = quaternion_to_matrix(quaternions)
        # rotate xyz
        self._xyz = torch.matmul(self._xyz, rotation_matrix.T)
        # rotate gaussian quaternions
        self._rotation = torch.nn.functional.normalize(quat_multiply(
            self._rotation,
            quaternions,
        ))
        self._features = transform_shs(self._features, rotation_matrix)

    def numpy(self):
        return {
            "xyz": self._xyz.cpu().numpy(),
            "scaling": torch.log(self._scaling).cpu().numpy(),
            "rotation": self._rotation.cpu().numpy(),
            "opacity": torch.logit(self._opacity).cpu().numpy(),
            "features": self._features.cpu().numpy(),
        }
        
def construct_from_ply(ply_path: str, device: torch.device = torch.device("cuda:0")):
    plydata = PlyData.read(ply_path)

    xyz = np.stack((
        np.asarray(plydata.elements[0]["x"]),
        np.asarray(plydata.elements[0]["y"]),
        np.asarray(plydata.elements[0]["z"]),
    ), axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    features_rest = load_array_from_plyelement(plydata.elements[0], "f_rest_").reshape((xyz.shape[0], 3, -1))
    # auto determine sh_degrees
    features_rest_dims = features_rest.shape[-1]
    for i in range(4):
        if features_rest_dims == (i + 1) ** 2 - 1:
            sh_degrees = i
            break
    assert sh_degrees >= 0, f"invalid sh_degrees={sh_degrees}"
    # N x 3 x 1 + N x 3 x 15 --> N x 3 x 16 --> N x 16 x 3
    features = np.transpose(np.concatenate([features_dc, features_rest], axis=2), (0, 2, 1)) 
    scales = load_array_from_plyelement(plydata.elements[0], "scale_")
    rots = load_array_from_plyelement(plydata.elements[0], "rot_")
    return GaussianModel(
        xyz=xyz,
        opacity=opacities,
        features=features,
        scaling=scales,
        rotation=rots,
        device=device,
        sh_degrees=sh_degrees,
    )

def save_to_ply(gaussian: GaussianModel, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ply_gs = gaussian.numpy()
    
    xyz, scaling, rotation, opacity, features = ply_gs["xyz"], ply_gs["scaling"], ply_gs["rotation"], ply_gs["opacity"], ply_gs["features"],
    normals = np.zeros_like(xyz)
    features = np.transpose(features, (0, 2, 1))
    features_dc = features[:, :, :1]
    features_rest = features[:, :, 1:]
    f_dc = features_dc.reshape((features_dc.shape[0], -1))
    if gaussian.sh_degrees > 0:
        f_rest = features_rest.reshape((features_rest.shape[0], -1))
    else:
        f_rest = np.zeros((f_dc.shape[0], 0))
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(features_dc.shape[1] * features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    if gaussian.sh_degrees > 0:
        for i in range(features_rest.shape[1] * features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotation.shape[1]):
        l.append('rot_{}'.format(i))

    dtype_full = [(attribute, 'f4') for attribute in l]
    attribute_list = [xyz, normals, f_dc, f_rest, opacity, scaling, rotation]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(attribute_list, axis=1)
    # do not save 'features_extra' for ply
    # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, f_extra), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
        
mat_sh1 = wp.types.matrix(shape=(3, 3), dtype=wp.float32)
mat_sh2 = wp.types.matrix(shape=(5, 3), dtype=wp.float32)
mat_sh3 = wp.types.matrix(shape=(7, 3), dtype=wp.float32)
mat_D1 = wp.types.matrix(shape=(3, 3), dtype=wp.float32)
mat_D2 = wp.types.matrix(shape=(5, 5), dtype=wp.float32)
mat_D3 = wp.types.matrix(shape=(7, 7), dtype=wp.float32)

@wp.kernel
def transform_kernel(
    _xyz_init: wp.array(dtype=wp.vec3),
    _rotation_init: wp.array(dtype=wp.quat),
    _features_init_sh1: wp.array(dtype=mat_sh1),
    _features_init_sh2: wp.array(dtype=mat_sh2),
    _features_init_sh3: wp.array(dtype=mat_sh3),
    _r_xyzws: wp.array(dtype=wp.quat),
    _t_xyzs: wp.array(dtype=wp.vec3),
    D_1: wp.array(dtype=mat_D1),
    D_2: wp.array(dtype=mat_D2),
    D_3: wp.array(dtype=mat_D3),
    _env_idxs: wp.array(dtype=wp.int64),
    _asset_idxs: wp.array(dtype=wp.int64),
    _xyz: wp.array(dtype=wp.vec3),
    _rotation: wp.array(dtype=wp.vec4),
    _features_sh1: wp.array(dtype=mat_sh1),
    _features_sh2: wp.array(dtype=mat_sh2),
    _features_sh3: wp.array(dtype=mat_sh3),
    num_envs: wp.int64,
):
    tid = wp.tid()
    env_id = _env_idxs[tid]
    asset_id = _asset_idxs[tid]
    transform_idx = int(asset_id * num_envs + env_id)
    r_xyzw = _r_xyzws[transform_idx]
    t_xyz = _t_xyzs[transform_idx]
    input_xyz = _xyz_init[tid]
    input_rotation = _rotation_init[tid]
    _xyz[tid] = wp.quat_rotate(r_xyzw, input_xyz) + t_xyz
    output_rotation = wp.normalize(wp.mul(r_xyzw, input_rotation))
    _rotation[tid] = wp.vec4(output_rotation[3], output_rotation[0], output_rotation[1], output_rotation[2])
    _features_sh1[tid] = wp.mul(D_1[transform_idx], _features_init_sh1[tid])
    _features_sh2[tid] = wp.mul(D_2[transform_idx], _features_init_sh2[tid])
    _features_sh3[tid] = wp.mul(D_3[transform_idx], _features_init_sh3[tid])

class FastGaussianModelManager:
    def __init__(
                self, 
                gaussian_models: list[GaussianModel], 
                num_envs: int = 1, 
                device: torch.device = torch.device("cuda:0"),
                active_sh_degree: int = 3,
                ):
        super().__init__()
        self.models = gaussian_models
        self.device = device
        self.num_envs = num_envs
        # calculate total gaussian num
        total_gaussian_num = 0
        model_gaussian_indices = []
        for i in gaussian_models:
            n = i.get_xyz.shape[0] * self.num_envs
            model_gaussian_indices.append((total_gaussian_num, total_gaussian_num + n))
            total_gaussian_num += n
        self.model_gaussian_indices = np.array(model_gaussian_indices, dtype=int)
        self.active_sh_degree = active_sh_degree
        
        self._xyz_init = torch.zeros((total_gaussian_num, 3), dtype=torch.float32)
        self._opacity_init = torch.zeros((total_gaussian_num, 1), dtype=torch.float32)
        self._features_init = torch.zeros([total_gaussian_num] + list(gaussian_models[0].get_features.shape[1:]), dtype=torch.float32)
        self._scaling_init = torch.zeros((total_gaussian_num, 3), dtype=torch.float32)
        self._rotation_init = torch.zeros((total_gaussian_num, 4), dtype=torch.float32)
        self._env_idxs = torch.zeros((total_gaussian_num, ), dtype=torch.int64)
        self._asset_idxs = torch.zeros((total_gaussian_num,), dtype=torch.int64)
        # merge gaussians
        for idx, model in enumerate(gaussian_models):
            begin, end = self.model_gaussian_indices[idx].tolist()
            self._xyz_init[begin:end] = model.get_xyz.repeat([self.num_envs] + [1] * (len(model.get_xyz.size()) - 1))
            self._opacity_init[begin:end] = model.get_opacity.repeat([self.num_envs] + [1] * (len(model.get_opacity.size()) - 1))
            self._features_init[begin:end] = model.get_features.repeat([self.num_envs] + [1] * (len(model.get_features.size()) - 1))
            self._scaling_init[begin:end] = model.get_scaling.repeat([self.num_envs] + [1] * (len(model.get_scaling.size()) - 1))
            self._rotation_init[begin:end] = model.get_rotation.repeat([self.num_envs] + [1] * (len(model.get_rotation.size()) - 1))
            env_idxs = torch.arange(self.num_envs).unsqueeze(0).repeat((model.get_xyz.size()[0], 1)).T.flatten()
            self._env_idxs[begin:end] = env_idxs
            self._asset_idxs[begin:end] = idx
        self._xyz = self._xyz_init.to(device)
        self._opacity = self._opacity_init.to(device)
        self._features = self._features_init.to(device)
        self._scaling = self._scaling_init.to(device)
        self._rotation = self._rotation_init.to(device)
        self.gs_num = self._xyz.size()[0]
        self._xyz_init_wp = wp.from_torch(self._xyz_init.to(self.device), dtype=wp.vec3)
        self._rotation_init_wp = wp.from_torch(self._rotation_init[:, [1,2,3,0]].to(self.device), dtype=wp.quat)
        self._features_init_sh1_wp = wp.from_torch(self._features_init[:, 1:4, :].to(self.device), dtype=mat_sh1)
        self._features_init_sh2_wp = wp.from_torch(self._features_init[:, 4:9, :].to(self.device), dtype=mat_sh2)
        self._features_init_sh3_wp = wp.from_torch(self._features_init[:, 9:16, :].to(self.device), dtype=mat_sh3)
        self._env_idxs_wp = wp.from_torch(self._env_idxs.to(self.device), dtype=wp.int64)
        self._asset_idxs_wp = wp.from_torch(self._asset_idxs.to(self.device), dtype=wp.int64)
        self._xyz_wp = wp.from_torch(self._xyz, dtype=wp.vec3)
        self._rotation_wp = wp.from_torch(self._rotation, dtype=wp.vec4)
        self._features_sh1_wp = wp.from_torch(self._features_init[:, 1:4, :].to(self.device), dtype=mat_sh1)
        self._features_sh2_wp = wp.from_torch(self._features_init[:, 4:9, :].to(self.device), dtype=mat_sh2)
        self._features_sh3_wp = wp.from_torch(self._features_init[:, 9:16, :].to(self.device), dtype=mat_sh3)
        
    def get_model_gaussian_indices(self, idx: int):
        return self.model_gaussian_indices[idx].tolist()

    def get_model(self, idx: int) -> GaussianModel:
        return self.models[idx]

    def transform_with_vectors(
            self,
            r_wxyzs: torch.tensor, # (num_assets, num_envs, 4)
            t_xyzs: torch.tensor, # (num_assets, num_envs, 3)
    ):
        '''
        Fast object transformation. For simulation.
        '''
        r_wxyzs_device = r_wxyzs.float().to(self.device)
        t_xyzs_device = t_xyzs.float().to(self.device)
        rotation_matrix = quaternion_to_matrix(r_wxyzs_device)
        permuted_rotation_matrix = rotation_matrix[:, :, [1, 2, 0]][:, [1, 2, 0], :]
        rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix)
        D_1 = wp.from_torch(wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2], self.device), dtype=mat_D1)
        D_2 = wp.from_torch(wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2], self.device), dtype=mat_D2)
        D_3 = wp.from_torch(wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2], self.device), dtype=mat_D3)
        _r_xyzws = wp.from_torch(r_wxyzs_device[:, [1,2,3,0]], dtype=wp.quat)
        _t_xyzs = wp.from_torch(t_xyzs_device, dtype=wp.vec3)
        wp.launch(
            kernel=transform_kernel,
            dim=self.gs_num,
            inputs=[
                self._xyz_init_wp, 
                self._rotation_init_wp, 
                self._features_init_sh1_wp,
                self._features_init_sh2_wp,
                self._features_init_sh3_wp,
                _r_xyzws, 
                _t_xyzs,
                D_1,
                D_2,
                D_3,
                self._env_idxs_wp,
                self._asset_idxs_wp,
                self._xyz_wp,
                self._rotation_wp,
                self._features_sh1_wp,
                self._features_sh2_wp,
                self._features_sh3_wp,
                wp.int64(self.num_envs),
            ],
        )
        self._xyz = wp.to_torch(self._xyz_wp)
        self._rotation = wp.to_torch(self._rotation_wp)
        self._features[:, 1:4, :] = wp.to_torch(self._features_sh1_wp)
        self._features[:, 4:9, :] = wp.to_torch(self._features_sh2_wp)
        self._features[:, 9:16, :] = wp.to_torch(self._features_sh3_wp)
        
    @property
    def get_scaling(self):
        return self._scaling

    @property
    def get_rotation(self):
        return self._rotation

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return self._features

    @property
    def get_opacity(self):
        return self._opacity