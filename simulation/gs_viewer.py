import os
import torch
import sys
import numpy as np
from urdfpy import URDF
import viser
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Literal, List
import cv2
sys.path.insert(0, os.getcwd())
from simulation.fast_gaussian_model_manager import FastGaussianModelManager, construct_from_ply, matrix_to_quaternion
from simulation.utils.gs_viewer_utils import (
    ClientThread,
    ViewerRenderer,
    GSPlatRenderer,
    render_meshes_depth_pytorch3d,
)
from simulation.utils.gs_viewer_utils import *
import imageio
from simulation.utils.constants import FR3_DEFAULT_CFG
import pytorch_kinematics as pk
import trimesh
import copy

def render_and_save_specific_view(renderer, device, file_path, cfg, R, T, verbose=False, render_depth=False, render_alpha=False, save=True, return_outputs=False, return_torch=False):
    fx = cfg["fx"]
    fy = cfg["fy"]
    cx = cfg["cx"]
    cy = cfg["cy"]
    image_width = cfg["image_width"]
    image_height = cfg["image_height"]
    R = torch.tensor(R, dtype=torch.float, device=device)
    T = torch.tensor(T, dtype=torch.float, device=device)
    camera = Cameras(
        R=R.unsqueeze(0),
        T=T.unsqueeze(0),
        fx=torch.tensor([fx], device=device),
        fy=torch.tensor([fy], device=device),
        cx=torch.tensor([cx], device=device),
        cy=torch.tensor([cy], device=device),
        width=torch.tensor([image_width], device=device),
        height=torch.tensor([image_height], device=device),
        appearance_id=torch.tensor([0], device=device),
        normalized_appearance_id=torch.tensor([0.0], device=device),
        time=torch.tensor([0.0], device=device),
        distortion_params=None,
        camera_type=torch.tensor([0], device=device),
    )[0].to_device(device)

    with torch.no_grad():
        output = renderer.get_outputs(camera, render_depth=render_depth)
        rendered_image = output["render"]
        if return_torch:
            rendered_image = rendered_image.clamp(0, 1).permute(1, 2, 0)
            rendered_image = (rendered_image * 255).to(torch.uint8)
        else:
            rendered_image = rendered_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            rendered_image = (rendered_image * 255).astype('uint8')
        if render_depth:
            if return_torch:
                depth_image = output["depth"][0]
                depth_image = torch.clamp(depth_image * 1000, 0, 65535).to(torch.uint16)
            else:
                depth_image = output["depth"][0].cpu().numpy()
                depth_image = np.clip(depth_image * 1000, 0, 65535).astype('uint16')
        if render_alpha:
            if return_torch:
                alpha_image = output["alpha"]
            else:
                alpha_image = output["alpha"].cpu().numpy()
                
    if save:
        if file_path.lower().endswith(".jpg"):
            imageio.imwrite(file_path, rendered_image[:, :, :3], quality=100)
        else:
            imageio.imwrite(file_path, rendered_image)
        if render_depth:
            depth_output_path = os.path.splitext(file_path)[0] + "_depth.png"
            cv2.imwrite(depth_output_path, depth_image)
        if verbose:
            print(f"saved: {file_path}")
    if return_outputs:
        return {
            "rgb_image": rendered_image,
            "depth_image": depth_image if render_depth else None,
            "alpha_image": alpha_image if render_alpha else None,
        }
    else:
        return
    
def _load_rigid_from_file(load_from: str, device: torch.device=torch.device("cuda:0"), filename="object_3dgs_abs.ply"):
    model = construct_from_ply(
        ply_path=Path(load_from).parent / filename, 
        device=device,
    )
    if "scene" in filename:
        mesh_path = "scene_gs_tsdf_fusion_post_abs_reduced.obj"
    else:
        mesh_path = "mesh_w_vertex_color_abs.obj"
    mesh_model_path= Path(load_from).parent / mesh_path
    mesh_model = trimesh.load(mesh_model_path)
    return model, mesh_model

def _load_articulation_from_file(load_from: str, device: torch.device=torch.device("cuda:0")):
    urdf_robot = URDF.load(load_from)
    fk = urdf_robot.link_fk()
    models = []
    part_names = []
    mesh_models = []
    for urdf_link, transmat in fk.items():
        part_name = urdf_link.name
        model = construct_from_ply(
            ply_path=Path(load_from).parent / "parts" / f"{part_name}_3dgs_abs.ply", 
            device=device,
        )
        model.translate(torch.from_numpy(-transmat[:3, 3]).to(model._xyz.dtype).to(device))
        model.rotate(matrix_to_quaternion(torch.from_numpy(np.linalg.inv(transmat[:3, :3])).to(model._rotation.dtype).to(device)))
        models.append(model)
        part_names.append(part_name)
        # merge all visual meshes for this link
        mesh_model = trimesh.util.concatenate([trimesh.util.concatenate([mesh.apply_transform(visual.origin) for mesh in visual.geometry.meshes]) for visual in urdf_link.visuals])
        mesh_models.append(mesh_model)
    return models, part_names, urdf_robot, mesh_models

def _load_robot_from_file(load_from: str, device: torch.device=torch.device("cuda:0")):
    urdf_robot = URDF.load(load_from)
    fk = urdf_robot.link_fk(FR3_DEFAULT_CFG)
    models = []
    part_names = []
    mesh_models = []
    for urdf_link, transmat in fk.items():
        part_name = urdf_link.name
        if part_name == "fr3_link8":
            continue
        if part_name == "fr3_hand_tcp" or "sc" in part_name:
            continue
        model = construct_from_ply(
            ply_path=Path(load_from).parent / "parts" / f"{part_name}_3dgs_abs.ply", 
            device=device,
        )
        model.translate(torch.from_numpy(-transmat[:3, 3]).to(model._xyz.dtype).to(device))
        model.rotate(matrix_to_quaternion(torch.from_numpy(np.linalg.inv(transmat[:3, :3])).to(model._rotation.dtype).to(device)))
        models.append(model)
        part_names.append(part_name)
        # merge all visual meshes for this link
        mesh_model = trimesh.util.concatenate([trimesh.util.concatenate([mesh.apply_transform(visual.origin) for mesh in visual.geometry.meshes]) for visual in urdf_link.visuals])
        mesh_models.append(mesh_model)
    return models, part_names, urdf_robot, mesh_models

class GenesisGaussianViewer:
    def __init__(
        self, 
        genesis_scene_path_dict: dict, 
        genesis_scene_dict: dict,
        device: torch.device=torch.device("cuda:0"),
        host: str = "0.0.0.0",
        port: int = 7007,
        background_color: Tuple = (0.8, 0.8, 0.8),
        image_format: Literal["jpeg", "png"] = "jpeg",
        active_sh_degree: int = 3,
        num_envs: int = 1,
        args_cli = None,
        render_depth = False,
    ):
        super().__init__()
        self.genesis_scene_path_dict = genesis_scene_path_dict
        self.genesis_scene_dict = genesis_scene_dict
        self.device = device
        self.asset_names = []
        self.gs_model_list = []
        self.part_names = []
        self.args_cli = args_cli
        self.num_envs = num_envs
        self.urdfs = dict()
        self.render_depth = render_depth
        if render_depth:
            self._mesh_verts_list = []
            self._mesh_faces_list = []
            self._mesh_matrices = []
        for asset_name, asset_config in tqdm(self.genesis_scene_path_dict.items(), desc="Loading 3DGS assets ..."):
            if "robot" in asset_name:
                gs_models, part_names, urdf_robot, mesh_models = _load_robot_from_file(asset_config, device=torch.device("cpu"))
                self.gs_model_list.extend(gs_models)
                self.part_names.extend(part_names)
                self.asset_names.extend([asset_name+f"_{part_name}" for part_name in part_names])
                if render_depth:
                    for mesh_model in mesh_models:
                        self._mesh_verts_list.append(np.asarray(mesh_model.vertices, dtype=np.float32))
                        self._mesh_faces_list.append(np.asarray(mesh_model.faces, dtype=np.int64))
                        self._mesh_matrices.append(np.eye(4, dtype=np.float32))
                self.urdfs[asset_name] = pk.build_chain_from_urdf(open(asset_config, mode="rb").read()).to(device=torch.device(device))
            else:
                urdf_path = asset_config
                if "_abs.urdf" in urdf_path:  # detected articulated object
                    print(f"Detected articulated object: {urdf_path}")
                    gs_models, part_names, urdf_robot, mesh_models = _load_articulation_from_file(urdf_path, device=torch.device("cpu"))
                    self.gs_model_list.extend(gs_models)
                    self.part_names.extend(part_names)
                    self.asset_names.extend([asset_name+f"_{part_name}" for part_name in part_names])
                    if render_depth:
                        for mesh_model in mesh_models:
                            self._mesh_verts_list.append(np.asarray(mesh_model.vertices, dtype=np.float32))
                            self._mesh_faces_list.append(np.asarray(mesh_model.faces, dtype=np.int64))
                            self._mesh_matrices.append(np.eye(4, dtype=np.float32))
                    self.urdfs[asset_name] = pk.build_chain_from_urdf(open(urdf_path, mode="rb").read()).to(device=torch.device(device))
                else:
                    gs_model, mesh_model = _load_rigid_from_file(urdf_path, device=torch.device("cpu"), 
                                                    filename="scene_gs_abs.ply" if "background" in asset_name \
                                                    else "object_3dgs_abs.ply")
                    self.gs_model_list.append(gs_model)
                    self.part_names.append(None)
                    self.asset_names.append(asset_name)
                    if render_depth:
                        self._mesh_verts_list.append(np.asarray(mesh_model.vertices, dtype=np.float32))
                        self._mesh_faces_list.append(np.asarray(mesh_model.faces, dtype=np.int64))
                        self._mesh_matrices.append(np.eye(4, dtype=np.float32))
        self.env_gs_model_manager = FastGaussianModelManager(
            gaussian_models=self.gs_model_list, 
            num_envs=self.num_envs,
            device=device,
            active_sh_degree=active_sh_degree,
        )
        self.renderer = GSPlatRenderer()
        self.host = host
        self.port = port
        self.background_color = background_color
        self.image_format = image_format
        self.viewer_renderer = ViewerRenderer(
            self.env_gs_model_manager,
            self.renderer,
            torch.tensor(self.background_color, dtype=torch.float, device=self.device),
        )
        self.clients = {}
        self.server = None
    def update(self):
        with torch.no_grad():
            # equivalent rotation and translation
            # TODO: currently only supports num_envs = 1
            r_wxyzs = []
            t_xyzs = []

            fk_results = dict()
            for genesis_asset_name in self.genesis_scene_dict.keys():
                if genesis_asset_name in self.urdfs:
                    joint_pos = self.genesis_scene_dict[genesis_asset_name].get_dofs_position()
                    ret = self.urdfs[genesis_asset_name].forward_kinematics(joint_pos.to(self.device))
                    fk_results[genesis_asset_name] = ret
            for idx, names in enumerate(zip(self.asset_names, self.part_names)):
                asset_name, urdf_part_name = names
                if urdf_part_name is not None:
                    genesis_asset_name = asset_name.replace(f"_{urdf_part_name}", "")
                    part_name = urdf_part_name.replace(f"fr3_", "")  # remove "fr3_" prefix
                    part_name = part_name.replace(f"finger", "_finger")  # avoid naming conflict 
                    matrix_local = fk_results[genesis_asset_name][urdf_part_name].get_matrix()
                    quat = self.genesis_scene_dict[genesis_asset_name].get_quat().to(self.device)
                    pos = self.genesis_scene_dict[genesis_asset_name].get_pos().to(self.device)
                    matrix_world = torch.eye(4).to(self.device)
                    matrix_world[:3, :3] = pk.quaternion_to_matrix(quat)
                    matrix_world[:3, 3] = pos
                    matrix = matrix_world @ matrix_local
                    if self.render_depth:
                        self._mesh_matrices[idx] = matrix[0].cpu().numpy()
                    r_wxyz = pk.matrix_to_quaternion(matrix[0, :3, :3])
                    t_xyz = matrix[0, :3, 3]
                    r_wxyz = r_wxyz.unsqueeze(0)
                    t_xyz = t_xyz.unsqueeze(0)
                else:
                    if "background" in asset_name:
                        r_wxyz = torch.tensor([1.0, 0.0, 0.0, 0.0]).to(torch.float32).to(self.device)
                        t_xyz = torch.tensor([0.0, 0.0, 0.0]).to(torch.float32).to(self.device)
                    else:
                        r_wxyz = self.genesis_scene_dict[asset_name].get_quat().to(self.device)
                        t_xyz = self.genesis_scene_dict[asset_name].get_pos().to(self.device)
                    if self.render_depth:
                        matrix_world = torch.eye(4).to(self.device)
                        matrix_world[:3, :3] = pk.quaternion_to_matrix(r_wxyz)
                        matrix_world[:3, 3] = t_xyz
                        self._mesh_matrices[idx] = matrix_world.cpu().numpy()
                    r_wxyz = r_wxyz.unsqueeze(0)
                    t_xyz = t_xyz.unsqueeze(0)
                r_wxyzs.append(r_wxyz)
                t_xyzs.append(t_xyz)
            r_wxyzs = torch.cat(r_wxyzs, 0)
            t_xyzs = torch.cat(t_xyzs, 0)
            
            self.env_gs_model_manager.transform_with_vectors(
                r_wxyzs=r_wxyzs,
                t_xyzs=t_xyzs)
            if self.server is not None:
                self.rerender_for_all_client()

    def render_depth_for_frame(self, w2c: np.ndarray, camera_intr: dict, device: torch.device, znear: float = 0.1, zfar: float = 1000.0, background_depth: float = 10.0) -> np.ndarray:
        """Render current mesh depth image using PyTorch3D. w2c is either 4x4 or R(3,3)+T(3,) for world to camera."""
        if not self.render_depth or not self._mesh_verts_list:
            return np.full((camera_intr["image_height"], camera_intr["image_width"]), background_depth, dtype=np.float32)
        return render_meshes_depth_pytorch3d(
            self._mesh_verts_list,
            self._mesh_faces_list,
            self._mesh_matrices,
            w2c,
            camera_intr,
            device,
            znear=znear,
            zfar=zfar,
            background_depth=background_depth,
        )

    def start(self, block: bool=False):
        # create viser server
        self.server = viser.ViserServer(host=self.host, port=self.port)
        self.server.configure_theme(
            control_layout="collapsible",
            show_logo=False,
        )
        with self.server.add_gui_folder("Env"):
            self.env_id_modifier = self.server.add_gui_slider(
                "Env ID",
                min=-1,
                max=(self.num_envs - 1),
                step=1,
                initial_value=-1,
            )
            self.env_id_modifier.on_update(self._handle_option_updated)
        # register hooks
        self.server.on_client_connect(self._handle_new_client)
        self.server.on_client_disconnect(self._handle_client_disconnect)

        if block:
            while True:
                time.sleep(999)
    
    def _handle_new_client(self, client: viser.ClientHandle) -> None:
        """
        Create and start a thread for every new client
        """

        # create client thread
        client_thread = ClientThread(self, self.viewer_renderer, client)
        client_thread.start()
        # store this thread
        self.clients[client.client_id] = client_thread

    def _handle_client_disconnect(self, client: viser.ClientHandle):
        """
        Destroy client thread when client disconnected
        """
        try:
            self.clients[client.client_id].stop()
            del self.clients[client.client_id]
        except Exception as err:
            print(err)

    def _handle_option_updated(self, _):
        """
        Push new render to all clients
        """
        return self.rerender_for_all_client()

    def handle_option_updated(self, _):
        return self._handle_option_updated(_)

    def rerender_for_client(self, client_id: int):
        """
        Render for specific client
        """
        try:
            # Switch to low resolution mode first, then notify the client to render
            self.clients[client_id].state = "low"
            self.clients[client_id].render_trigger.set()
        except:
            # ignore errors
            pass
        
    def rerender_for_all_client(self):
        for i in self.clients:
            self.rerender_for_client(i)


def init_gui(gs_viewer: GenesisGaussianViewer, robot, jnt_names, default_joint_pos):
    with gs_viewer.server.add_gui_folder("Joints"):
        gs_viewer.joint_modifiers = dict()
        for i, joint_name in enumerate(jnt_names[:7]):
            joint_limits = robot.get_dofs_limit([i])
            gs_viewer.joint_modifiers[robot.get_joint(name=joint_name).name] = gs_viewer.server.add_gui_slider(
                robot.get_joint(name=joint_name).name,
                min=float(joint_limits[0].item()),
                max=float(joint_limits[1].item()),
                step=np.pi / 180,
                initial_value=float(default_joint_pos[i]),
            )
            gs_viewer.joint_modifiers[robot.get_joint(name=joint_name).name].on_update(gs_viewer._handle_option_updated)
        gs_viewer.joint_modifiers["fr3_finger_joint"] = gs_viewer.server.add_gui_slider(
            "fr3_finger_joint",
            min=0,
            max=0.04,
            step=0.01,
            initial_value=0.04
        )
        gs_viewer.joint_modifiers["fr3_finger_joint"].on_update(gs_viewer._handle_option_updated)
