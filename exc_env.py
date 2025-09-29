# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import time
import torch
from isaaclab.sim import SimulationContext
from isaacsim.core.utils.prims import get_prim_at_path
import numpy as np
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom
import isaaclab.sim as sim_utils
from isaaclab.sim.schemas import ArticulationRootPropertiesCfg, modify_articulation_root_properties
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCfg, RigidObject, AssetBaseCfg, AssetBase
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg, SubTerrainBaseCfg, TerrainImporter
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform, axis_angle_from_quat, quat_from_angle_axis
from scipy.spatial.transform import Rotation as R
from torch.utils.tensorboard import SummaryWriter
import omni.physics.tensors as tensors
import matplotlib.pyplot as plt


@configclass
class ExcEnvCfg(DirectRLEnvCfg):
    # env
    n_steps=49
    episode_length_s = 19.95  # 500 timesteps
    decimation = 9
    action_space = 3
    observation_space = 26
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/60,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=3, env_spacing=20.0, replicate_physics=True)
    
    # robot
    robot = ArticulationCfg(
        collision_group = 0,
        prim_path="/World/envs/env_.*/EC300E_NLC_MONO",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/volvo/Downloads/exc.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=ArticulationRootPropertiesCfg(
                fix_root_link=True, enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
            #"swing_joint": 0.0,
            "boom_joint": 0.0,
            "arm_joint": 0.0,
            "buck_joint": 0.0,
            },
            pos=(0.0, 0.0, 1.),
            rot=(0.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            # "swing_act": ImplicitActuatorCfg(
            #     joint_names_expr=["swing_joint"],
            #     effort_limit_sim=100.0,
            #     velocity_limit_sim=100.0,
            #     stiffness=10000.0,
            #     damping=100.0,
            # ),
            "boom_act": ImplicitActuatorCfg(
                joint_names_expr=["boom_joint"],
                effort_limit_sim=1e9,
                velocity_limit_sim=1e9,
                stiffness=0.0,
                damping=0.0,
            ),
            "arm_act": ImplicitActuatorCfg(
                joint_names_expr=["arm_joint"],
                effort_limit_sim=1e9,
                velocity_limit_sim=1e9,
                stiffness=0.0,
                damping=0.0,
            ),
            "bucket_act": ImplicitActuatorCfg(
                joint_names_expr=["buck_joint"],
                effort_limit_sim=1e9,
                velocity_limit_sim=1e9,
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )

    # #ground plane
    # terrain = TerrainImporterCfg(
    #     #usd_path = f"/home/volvo/Downloads/plane.usd",
    #     #prim_path="/World/Plane",
    #     terrain_type="usd",
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #     ),
    #     num_envs = 3,
    #     env_spacing = 20,
    # )
    
    # object_cfg: RigidObjectCfg = A(
    #     prim_path="/World/envs/env_.*/Plane",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"/home/volvo/Downloads/plane.usd",
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             kinematic_enabled=True,
    #             disable_gravity=True,
    #             enable_gyroscopic_forces=False,
    #             # solver_position_iteration_count=8,
    #             # solver_velocity_iteration_count=0,
    #             # sleep_threshold=0.005,
    #             # stabilization_threshold=0.0025,
    #             # max_depenetration_velocity=0.,
    #         ),
    #         #mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -1.2), rot=(1.0, 0.0, 0.0, 0.0)),
    # )
    
    # object_cfg: AssetBaseCfg = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Plane",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"/home/volvo/Downloads/planee.usd",
    #         # rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #         #     kinematic_enabled=True,
    #         #     disable_gravity=True,
    #         #     enable_gyroscopic_forces=False,
    #         #     # solver_position_iteration_count=8,
    #         #     # solver_velocity_iteration_count=0,
    #         #     # sleep_threshold=0.005,
    #         #     # stabilization_threshold=0.0025,
    #         #     # max_depenetration_velocity=0.,
    #         # ),
    #         #mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
    #     #init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -1.2), rot=(1.0, 0.0, 0.0, 0.0)),
    # )
    
    # my_asset: AssetBaseCfg = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Plane",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -0.6], rot=[1.0, 0.0, 0.0, 0.0]),
    #     spawn=sim_utils.UsdFileCfg(usd_path=f"/home/volvo/Downloads/planee.usd"),
    # )
    
    my_asset: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Plane",  # applies to every sub-env
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, 0.0, -0.5],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/volvo/Downloads/planee.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                enable_gyroscopic_forces=True,
                # solver_position_iteration_count=8,
                # solver_velocity_iteration_count=0,
                # sleep_threshold=0.005,
                # stabilization_threshold=0.0025,
                # max_depenetration_velocity=0.,
            ),
        ),
    )
    
    action_scale = 7.5
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 1.5
    rot_reward_scale = 1.5
    open_reward_scale = 10.0
    action_penalty_scale = 0.05
    finger_reward_scale = 2.0


class ExcEnv(DirectRLEnv):

    cfg: ExcEnvCfg

    def __init__(self, cfg: ExcEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.joint_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.joint_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        
        self.tip_idx = self._robot.find_bodies("tip")[0][0]
        self.tip2_idx = self._robot.find_bodies("tip2")[0][0]
        self.tip3_idx = self._robot.find_bodies("tip3")[0][0]
        self.circle_mid_idx = self._robot.find_bodies("circle_mid")[0][0]
        self.cabin_idx = self._robot.find_bodies("UPPER_ASSY_EC300E_1")[0][0]
        self.boom_idx = self._robot.find_bodies("BOOM_EC300E_6_2_HD_1")[0][0]
        self.arm_idx = self._robot.find_bodies("ARM_EC300E_3_05_HD_1")[0][0]
        self.buck_idx = self._robot.find_bodies("BUCKET_EC300E_1_27_1")[0][0]
        self.base_idx = self._robot.find_bodies("base_link")[0][0]
        
        self.soil_height = sample_uniform(-2.0, -0.5, (self.num_envs), self.device)
        self.prev_error = torch.zeros((self.num_envs, 3), device=self.device)
        self.integral_error = torch.zeros((self.num_envs, 3), device=self.device)
        self.integral_error2 = torch.zeros((self.num_envs, 3), device=self.device)
        self.actions = torch.zeros((self.num_envs, 3), device=self.device)
        self.torque = torch.zeros((self.num_envs, 3), device=self.device)
        self.joint_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.v_max = torch.tensor([2,2,2], device=self.device)
        self.v_min = torch.tensor([2,2,2], device=self.device)
        self.plate_width, self.plate_height = 1.2, 0.94
        self.bucket_circle = 0.45
        self.bucket_vol = (self.bucket_circle * self.plate_height * self.plate_width) + (0.5 * torch.pi * (self.bucket_circle**2) * self.plate_width)
        self.pre_tip_pos = (self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.tip_idx] - self.scene.env_origins)[:,[0,2]]
        self.pre_cabin_pos = (self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.cabin_idx] - self.scene.env_origins)[:,[0,2]]
        self.pre_buck_pos = (self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.buck_idx] - self.scene.env_origins)[:,[0,2]]
        self.total_swept = torch.zeros(self.num_envs, device=self.device)
        self.fill_ratio = torch.zeros(self.num_envs, device=self.device)
        self.epi_step = torch.zeros(self.num_envs, device=self.device)
        self.reward_sum = torch.zeros(1, device=self.device)
        self.write_count = 0.
        self.max_step = self.cfg.episode_length_s / self.dt
        self.pre_fill_ratio = torch.zeros(self.num_envs, device=self.device)
        self.pre_tip_ang = torch.zeros(self.num_envs, device=self.device)
        self.pre_actions = torch.zeros((self.num_envs, 3), device=self.device)
        self.pre_joint_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ang_tip_plate = torch.zeros(self.num_envs, device=self.device)
        self.count_steps, self.tip_pos, self.tip_lin_vel, self.cabin_pos, self.base_pos, self.tip_pos2, self.tip_pos3, self.tip_ang, self.tip_ang2, self.circle_mid  = 0., 0., 0.,0.,0.,0.,0.,0., 0., 0.
        self.cabin_lin_vel, self.boom_ang, self.arm_ang, self.buck_ang = 0.,0.,0.,0.
        self.kj, self.nc1, self.nc2, self.nc3, self.nc4, self.nc5, self.nc6, self.nc7 = 0.,0.,0.,0.,0.,0.,0.,0.
        self.pc1, self.pc2, self.pc3, self.pc4, self.pc5 = 0.,0.,0.,0.,0.
        self.fill_terminal_reward, self.close_terminal_reward = 0., 0.
        
        self.tip_lin_vel = torch.zeros((self.num_envs, 2), device=self.device)
        self.tip_ang_vel = torch.zeros((self.num_envs), device=self.device)
        self.buck_lin_vel = torch.zeros((self.num_envs, 2), device=self.device)
        self.boom_pos = self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.boom_idx] - self.scene.env_origins
        self.arm_pos = self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.arm_idx] - self.scene.env_origins
        self.buck_pos = self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.buck_idx] - self.scene.env_origins
        # self.boom_pos = (self._robot.data.body_pos_w[0, self.boom_idx] - self.scene.env_origins)[:,[0,2]]
        # self.arm_pos = (self._robot.data.body_pos_w[0, self.arm_idx] - self.scene.env_origins)[:,[0,2]]
        # self.buck_pos = (self._robot.data.body_pos_w[0, self.buck_idx] - self.scene.env_origins)[:,[0,2]]
        #self.tip_pos1 = (self._robot.data.body_pos_w[0, self.tip_idx] - self.scene.env_origins[0])
        #self.boom_arm, self.arm_buck, self.buck_tip = self.exc_config(self.boom_pos, self.arm_pos, self.buck_pos, self.tip_pos1)
        self.fixed_pos = self._robot.data.root_pos_w.clone()
        self.obj_pos = self._robot.data.root_pos_w.clone()
        self.obj_pos[:,2] = self.obj_pos[:,2] - 1.274
        
        # soil param
        self.c = sample_uniform(0,105000, (self.num_envs), self.device)
        self.c_a = sample_uniform(0, self.c, (self.num_envs), self.device)
        self.sifa = sample_uniform(0.3, 0.8, (self.num_envs), self.device)
        self.uw = sample_uniform(17000, 22000, (self.num_envs), self.device)
        self.sbfa = sample_uniform(0.2, 0.4, (self.num_envs), self.device)
        self.cpf = sample_uniform(0, 300, (self.num_envs), self.device)
        
        # self.boom_limit = torch.tensor([-2e6, 2e6], device=self.device)
        # self.arm_limit = torch.tensor([-3e5, 3e5], device=self.device)
        # self.buck_limit = torch.tensor([-2e5, 2e5], device=self.device)
        self.boom_limit = torch.tensor([-2.0e6, 2.0e6], device=self.device)
        self.arm_limit = torch.tensor([-4.0e6, 4.0e6], device=self.device)
        self.buck_limit = torch.tensor([-2e5, 2e5], device=self.device)
        
        self.A_s = self.plate_width * self.plate_height # area of separation plate
        self.n = 5 # number of teeth
        self.alpha = 0.5 # tip semi angle
        self.tooth_r = 0.075 # tooth radius
        self.A_t = torch.pi * self.tooth_r**2
        self.B = self.plate_width # separation plate width
        self.max_L = torch.full_like(self.c, self.plate_height, dtype=torch.float32)
        self.soil_mass = torch.full_like(self.c, 0., dtype=torch.float32)
        self.P_cord = torch.zeros((self.num_envs, 2), device=self.device)
        self.centroid = torch.zeros((self.num_envs, 2), device=self.device)
        
        self.tau_base   = 0.   # (B,6)
        self.tau_joints = 0.   # (B,n)
        self.zero_actions = 0.
        self.vel_zero= 0.
        self.total_termination, self.total_positive, self.cf = 0., 0., 0.004
        self.ang, self.envs = 0., 0.
        
        self.writer = SummaryWriter(f"/home/volvo/soil_vs/runs/skrl")
        self.sim_view = tensors.create_simulation_view("torch")
        self.articulation_view = self.sim_view.create_articulation_view("/World/envs/env_*/EC300E_NLC_MONO/EC300E_NLC_MONO/base_link")
        self.st =0.
        
        
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["Dofbot"] = self._robot
    
        # self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        # self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        #self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # self.object = RigidObject(self.cfg.object_cfg)
        # self.scene.rigid_objects["object"] = self.object
        
        # assert_cfg = self.cfg.my_asset
        # assert_cfg.spawn.func(
        #     assert_cfg.prim_path, assert_cfg.spawn, translation=assert_cfg.init_state.pos, orientation=assert_cfg.init_state.rot,
        # )
        
        self.object = RigidObject(self.cfg.my_asset)
        self.scene.rigid_objects["object"] = self.object
        
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        

    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self.actions[:,0] = torch.clamp(self.actions[:,0]* 0.2, -0.2, 0.2)
        self.actions[:,1] = torch.clamp(self.actions[:,1]* 0.4, -0.4, 0.4)
        self.actions[:,2] = torch.clamp(self.actions[:,2]* 0.6, -0.6, 0.6)   
        
        # self.tip_pos = (self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.tip_idx] - self.scene.env_origins)[:,[0,2]]
        # self.tip_pos2 = (self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.tip2_idx] - self.scene.env_origins)[:,[0,2]]
        # self.tip_pos3 = (self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.tip3_idx] - self.scene.env_origins)[:,[0,2]]
        # self.circle_mid = (self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.circle_mid_idx] - self.scene.env_origins)[:,[0,2]]
        #self.tip_lin_vel = self._robot.data.body_lin_vel_w[list(range(0, self.num_envs)), self.tip_idx][:,[0,2]]
        #self.ang_tip_plate = self.tip_ang+torch.atan2(self.tip_lin_vel[:,1], self.tip_lin_vel[:,0])
        self.compute_fill_ratio(self.tip_pos)
        
        self.torque = self.compute_torque(self.actions, 0)
        f_joint, f_base, g_joint, g_base = self.soil_force()
        f_joint = torch.where(torch.sign(self.torque)==torch.sign(f_joint), 0., f_joint)
        f_joint = torch.where(torch.abs(self.torque)<torch.abs(f_joint), self.torque, f_joint)
        self.vel_zero = torch.where((torch.abs(self.torque)==torch.abs(f_joint)) & (self.torque != 0.))
        self.torque = self.torque + f_joint + g_joint
        self.zero_actions = self.actions.clone()
        self.zero_actions[self.vel_zero] = 0.
        
        
    # post-physics step calls    
    def _apply_action(self):
        # torque = self.compute_torque(self.zero_actions, 1)
        # self.torque[self.vel_zero] = torque[self.vel_zero]
        
        self.torque = self.compute_torque(self.actions, 1)
        
        # joint_pos = torch.cat((boom_ang, arm_ang, buck_ang),-1)
        # joint_vel = torch.zeros_like(joint_pos)
        # self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=list(range(0, self.num_envs)))
       
        # Clamp to torque limits
        self.torque[:,0] = torch.clamp(self.torque[:,0], min = self.boom_limit[0], max = self.boom_limit[1])
        self.torque[:,1] = torch.clamp(self.torque[:,1], min = self.arm_limit[0], max = self.arm_limit[1])
        self.torque[:,2] = torch.clamp(self.torque[:,2], min = self.buck_limit[0], max = self.buck_limit[1])
        
        #self.writer.add_scalar("tau_base", torch.max(self.tau_base), self.count_steps)
        # tau_base = f_base + g_base
        # self._robot.set_external_force_and_torque(tau_base[:,:3][:,None,:], tau_base[:,3:][:,None,:], body_ids=self.base_idx)
        #print(self.torque[self.torque.isnan().any(dim=-1)])
        
        self._robot.set_joint_effort_target(self.torque)
    

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        #cabin_lin_vel = self._robot.data.body_lin_vel_w[list(range(0, self.num_envs)), self.cabin_idx][:,[0,2]]
        self.circle_mid = (self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.circle_mid_idx] - self.scene.env_origins)[:,[0,2]]
        buck_pos = (self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.buck_idx] - self.scene.env_origins)[:,[0,2]]
        self.tip_pos = (self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.tip_idx] - self.scene.env_origins)[:,[0,2]]
        self.tip_pos2 = (self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.tip2_idx] - self.scene.env_origins)[:,[0,2]]
        self.tip_ang = self.batch_angle_xz(self.tip_pos2,self.tip_pos)
        #self.tip_lin_vel = self._robot.data.body_lin_vel_w[list(range(0, self.num_envs)), self.tip_idx][:,[0,2]]
        self.tip_lin_vel = (self.tip_pos - self.pre_tip_pos) / self.dt
        self.tip_ang_vel = (self.tip_ang - self.pre_tip_ang) / self.dt
        self.buck_lin_vel = (buck_pos - self.pre_buck_pos) / self.dt
        self.ang_tip_plate = torch.atan2(self.tip_lin_vel[:,1], self.tip_lin_vel[:,0]) - self.tip_ang
        self.compute_fill_ratio(self.tip_pos)
        self.pre_tip_pos = self.tip_pos.clone()
        self.pre_tip_ang = self.tip_ang.clone()
        self.pre_buck_pos = buck_pos.clone()
       
        self.cabin_pos = (self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.cabin_idx] - self.scene.env_origins)[:,[0,2]]
        cabin_lin_vel =  (self.cabin_pos - self.pre_cabin_pos) / self.dt
        self.pre_cabin_pos = self.cabin_pos.clone()
        joint_ang = self._robot.data.joint_pos
        self.boom_ang = joint_ang[:,0]
        self.arm_ang = joint_ang[:,1]
        self.buck_ang = joint_ang[:,2]
        
        self.joint_vel = (joint_ang - self.pre_joint_pos) / self.dt
        self.pre_joint_pos = joint_ang.clone()
        
        self.tip_ang2 = self.batch_angle_xz(self.tip_pos3,self.tip_pos)
        #self.tip_ang_vel = self._robot.data.body_ang_vel_w[list(range(0, self.num_envs)), self.tip_idx][:,1]
        # j = self.count_steps // self.cfg.n_steps
        # self.kj = 1 - torch.exp(torch.tensor(-0.01 * j))
        self.count_steps += 1
        self.epi_step += 1
        
        self.nc1 = torch.norm(self.tip_lin_vel, dim=-1) > 0.75
        self.nc2 = (self.ang_tip_plate < 0.) & ((self.tip_pos[:,1] - self.soil_height) < 0.)
        self.nc3 = torch.norm(cabin_lin_vel, dim=-1) > 0.1
        self.nc4 = (self.fill_ratio == 0.) & ((self.tip_pos[:,1] - self.soil_height) > 0.4)
        self.nc5 = (self.tip_pos[:,0] > -2.5) | (self.boom_ang <= self.joint_lower_limits[0]) | (self.boom_ang >= self.joint_upper_limits[0]) | (self.arm_ang <= self.joint_lower_limits[1]) | (self.arm_ang >= self.joint_upper_limits[1]) | (self.buck_ang <= self.joint_lower_limits[2]) | (self.buck_ang >= self.joint_upper_limits[2])
        #self.nc6 = (self.fill_ratio == 1.) & ((self.P_cord[:,1] <  self.soil_height) | (self.tip_pos[:,1] < self.soil_height)) 
        self.nc7 = (self.fill_ratio > 0.) & (self.tip_pos[:,1] > self.soil_height) & (self.tip_ang < 0.)
        
        # pc1 = self.fill_ratio > (self.kj * 0.68 + 0.3)
        # pc2 = torch.norm(self.cabin_pos - self.tip_pos, dim=-1) < 3.0
        # pc3 = (self.tip_pos[:,1] - self.soil_height) > (self.kj * 0.7 + 0.3)
        # pc4 = (self.tip_ang2 < 0.2) & (self.tip_ang2 > -0.1)
        # pc5 = buck_pos[:,1] > self.tip_pos[:,1]
        
        self.pc1 = self.fill_ratio >= (0.5 + 0.5 * self.cf)
        self.pc2 = torch.norm(self.cabin_pos - self.tip_pos, dim=-1) < 3.7 - 0.7 * self.cf
        self.pc3 = (self.tip_pos[:,1] - self.soil_height) > 0.3 + 0.7 * self.cf
        self.pc4 = (self.tip_ang2 < 0.2) & (self.tip_ang2 > -0.1)
        self.pc5 = self.fill_ratio >= (0.4 + 0.5 * self.cf)
        
        negative_terminations = torch.where(self.nc1 | self.nc2 | self.nc4 | self.nc5 | self.nc7, 1., 0.)
        #negative_terminations = torch.where(self.nc1 | self.nc2 | self.nc5 | self.nc7, 1., 0.)
        
        self.fill_terminal_reward = torch.where(self.pc1 & self.pc3 & self.pc4, 10., 0.)
        self.close_terminal_reward = torch.where(self.pc2 & self.pc3 & self.pc4 & (self.fill_ratio > 0.01), 5., 0.)
        
        positive_terminations = self.fill_terminal_reward + self.close_terminal_reward
        self.total_termination = self.total_termination + torch.count_nonzero(positive_terminations) + torch.count_nonzero(negative_terminations)
        self.total_positive = self.total_positive + torch.count_nonzero(positive_terminations)
        
        if self.count_steps % self.cfg.n_steps == 0.:
            self.cf = torch.where((self.total_positive / self.total_termination) > 0.5, self.cf + 0.004, self.cf)
            self.total_positive, self.total_termination = 0., 0.
       
        positive_terminations = torch.where((positive_terminations !=0.) | (self.max_step == self.epi_step), 1., 0.)
       
        # if self.count_steps % 1000 ==0:
        #     a=1.
            
        # else:
        #     a=0.    
        
        # return torch.tensor([a]*self.num_envs), torch.tensor([a]*self.num_envs)
        
        self.writer.add_scalar("nc1", torch.count_nonzero(self.nc1), self.count_steps)
        self.writer.add_scalar("nc2", torch.count_nonzero(self.nc2), self.count_steps)
        self.writer.add_scalar("nc3", torch.count_nonzero(self.nc3), self.count_steps)
        self.writer.add_scalar("nc4", torch.count_nonzero(self.nc4), self.count_steps)
        self.writer.add_scalar("nc5", torch.count_nonzero(self.nc5), self.count_steps)
        #self.writer.add_scalar("nc6", torch.count_nonzero(self.nc6), self.count_steps)
        self.writer.add_scalar("nc7", torch.count_nonzero(self.nc7), self.count_steps)
        
        return positive_terminations, negative_terminations

    def _get_rewards(self) -> torch.Tensor:
        # c1 = self.fill_ratio > (self.kj * 0.6 + 0.3)
        # c2 = torch.norm(self.cabin_pos - self.tip_pos, dim=-1) < 3.0
        # c3 = (self.tip_pos[:,1] - self.soil_height) > 1.0
        # c4 = (self.tip_ang2 < 0.2) & (self.tip_ang2 > -0.1)
        # c5 = torch.norm(self.tip_lin_vel, dim=-1) <= 0.4
        
        # move_down = torch.where((self.fill_ratio <0.05) & c5, -0.1*self.tip_lin_vel[:,1], 0.)
           
        # filling = torch.where(~c2 & c5, self.fill_ratio - self.pre_fill_ratio, 0.)
        # self.pre_fill_ratio = self.fill_ratio.clone()
        
        # move_up = torch.where(((c1 | c2) & ~c3) & c5, 0.1*self.tip_lin_vel[:,1], 0.)
        
        # curl = torch.where(((c1 | c2) & ~c4) & c5, 0.05*self.tip_ang_vel, 0.)
        
        move_down = torch.where((self.fill_ratio <0.01) & (~self.pc2), -0.1*self.tip_lin_vel[:,1], 0.)
           
        filling = torch.where(~(self.pc1 | self.pc2), self.fill_ratio - self.pre_fill_ratio, 0.)
        self.pre_fill_ratio = self.fill_ratio.clone()
        
        curl = torch.where(((self.pc2 | self.pc5) & (~self.pc4)), 0.1*self.joint_vel[:,2], 0.)
          
        smooth = -0.001 * torch.norm(self.actions - self.pre_actions, p=1, dim=-1)
        self.pre_actions = self.actions.clone()
        
        reward = move_down + filling + curl + smooth + self.fill_terminal_reward + self.close_terminal_reward
        reward = torch.where(self.nc1 | self.nc2 | self.nc4 | self.nc5 | self.nc7, -1., reward)
        
        #reward = torch.where(self.nc1 | self.nc4 | self.nc5 | self.nc7, -1., reward)
        #reward = torch.where(self.nc2, reward-1., reward)
        #reward = torch.where(self.nc1 | self.nc2 | self.nc5 | self.nc7, -1., reward)
        self.reward_sum = self.reward_sum + reward[0]
        
        #print("tip_pos", self.tip_pos[self.tip_pos.isnan().any(dim=-1)])
        
        return reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        
        print(len(env_ids))
  
        if torch.isin(env_ids, torch.tensor([0.], device=self.device))[0]:
            self.writer.add_scalar("epi_reward", self.reward_sum, self.write_count)
            self.reward_sum = torch.zeros(1, device=self.device)
            self.write_count+=1
        
        # robot state
        cab_ang = sample_uniform(-0.1, 0.1, len(env_ids), self.device)
        self.ang = quat_from_angle_axis(cab_ang, torch.tensor([0.0, 1.0, 0.0], device=self.device))
        
        #self._robot.write_root_pose_to_sim(torch.cat((self.fixed_pos[env_ids], self.ang), dim=-1), env_ids=env_ids)
        #self.object.write_root_pose_to_sim(torch.cat((self.obj_pos[env_ids], self.ang), dim=-1), env_ids=env_ids)
        self._robot.write_root_state_to_sim(torch.cat((self.fixed_pos[env_ids], self.ang, torch.zeros_like(self.fixed_pos[env_ids]), torch.zeros_like(self.fixed_pos[env_ids]),), dim=-1), env_ids=env_ids)
        self.object.write_root_state_to_sim(torch.cat((self.obj_pos[env_ids], self.ang, torch.zeros_like(self.obj_pos[env_ids]), torch.zeros_like(self.obj_pos[env_ids]),), dim=-1), env_ids=env_ids)
      
        self.scene.update(dt=self.physics_dt)
        copy_env_ids = env_ids.clone()
        # random = sample_uniform(0, 1, (len(env_ids)), self.device)
        # random_id = torch.where(random<0.25)[0]
        # copy_env_ids = env_ids[~torch.isin(env_ids, random_id)]
        # copy_env_ids1 = env_ids[random_id]
        # random_id1 = env_ids[random_id]
        
        
        while True:
            # boom_ang = sample_uniform(self.joint_lower_limits[0]+0.1, self.joint_upper_limits[0]-0.1, (len(copy_env_ids), 1), self.device)
            # arm_ang = sample_uniform(self.joint_lower_limits[1]+0.1, self.joint_upper_limits[1]-0.1, (len(copy_env_ids), 1), self.device)
            # buck_ang = sample_uniform(self.joint_lower_limits[2]+0.1, self.joint_upper_limits[2]-0.1, (len(copy_env_ids), 1), self.device)
            boom_ang = sample_uniform(self.joint_lower_limits[0]+0.1, -0.58, (len(copy_env_ids), 1), self.device)
            arm_ang = sample_uniform(self.joint_lower_limits[1]+0.1, -0.75, (len(copy_env_ids), 1), self.device)
            buck_ang = sample_uniform(self.joint_lower_limits[2]+0.1, -2.79, (len(copy_env_ids), 1), self.device)
            joint_pos = torch.cat((boom_ang, arm_ang, buck_ang),-1)
            joint_vel = torch.zeros_like(joint_pos)
            self._robot.set_joint_position_target(joint_pos, env_ids=copy_env_ids)
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=copy_env_ids)
    
            self._robot.write_root_state_to_sim(torch.cat((self.fixed_pos[env_ids], self.ang, torch.zeros_like(self.fixed_pos[env_ids]), torch.zeros_like(self.fixed_pos[env_ids]),), dim=-1), env_ids=env_ids)
            self.scene.update(dt=self.physics_dt)
            
            self.tip_pos = (self._robot.data.body_pos_w[copy_env_ids, self.tip_idx] - self.scene.env_origins[copy_env_ids])[:,[0,2]]
            condition = ((self.tip_pos[:, 1] - self.soil_height[copy_env_ids]) >= 0.4) | (self.tip_pos[:, 0] > -6.)
            copy_env_ids = copy_env_ids[condition]
        
            # self._robot.write_root_state_to_sim(torch.cat((self.fixed_pos[env_ids], self.ang, torch.zeros_like(self.fixed_pos[env_ids]), torch.zeros_like(self.fixed_pos[env_ids]),), dim=-1), env_ids=copy_env_ids)
            # self.scene.update(dt=self.physics_dt)
            if copy_env_ids.numel() == 0:
                break

        
        soil_in = torch.where((self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.tip_idx] - self.scene.env_origins)[:,2] < self.soil_height)    
        # while True:
        #     boom_ang = sample_uniform(self.joint_lower_limits[0]+0.1, self.joint_upper_limits[0]-0.1, (len(copy_env_ids1), 1), self.device)
        #     arm_ang = sample_uniform(self.joint_lower_limits[1]+0.1, self.joint_upper_limits[1]-0.1, (len(copy_env_ids1), 1), self.device)
        #     buck_ang = sample_uniform(self.joint_lower_limits[2]+0.1, self.joint_upper_limits[2]-0.1, (len(copy_env_ids1), 1), self.device)
        #     joint_pos = torch.cat((boom_ang, arm_ang, buck_ang),-1)
        #     joint_vel = torch.zeros_like(joint_pos)
        #     self._robot.set_joint_position_target(joint_pos, env_ids=copy_env_ids1)
        #     self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=copy_env_ids1)
        #     self.scene.update(dt=self.physics_dt)
            
        #     self.tip_pos = (self._robot.data.body_pos_w[copy_env_ids1, self.tip_idx] - self.scene.env_origins[copy_env_ids1])[:,[0,2]]
        #     condition = (self.tip_pos[:, 1] - self.soil_height[copy_env_ids1]) >= 0.
        #     copy_env_ids1 = copy_env_ids1[condition]
            
        #     if copy_env_ids1.numel() == 0:
        #         break
        
        self.soil_height[env_ids] = sample_uniform(-2.0, -0.5, (len(env_ids)), self.device)
        self.prev_error[env_ids] = torch.zeros((len(env_ids), 3), device=self.device)
        self.integral_error[env_ids] = torch.zeros((len(env_ids), 3), device=self.device)
        self.integral_error2[env_ids] = torch.zeros((len(env_ids), 3), device=self.device)
        self.actions[env_ids] = torch.zeros((len(env_ids), 3), device=self.device)
        self.pre_tip_pos[env_ids] = (self._robot.data.body_pos_w[env_ids, self.tip_idx] - self.scene.env_origins[env_ids])[:,[0,2]]
        self.pre_cabin_pos[env_ids] = (self._robot.data.body_pos_w[env_ids, self.cabin_idx] - self.scene.env_origins[env_ids])[:,[0,2]]
        self.pre_buck_pos[env_ids] = (self._robot.data.body_pos_w[env_ids, self.buck_idx] - self.scene.env_origins[env_ids])[:,[0,2]]
        self.total_swept[env_ids] = torch.zeros(len(env_ids), device=self.device)
        self.fill_ratio[env_ids] = torch.zeros(len(env_ids), device=self.device)
        self.pre_fill_ratio[env_ids] = torch.zeros(len(env_ids), device=self.device)
        # self.fill_ratio[random_id1] = sample_uniform(0., 1., (len(random_id1)), self.device)
        # self.pre_fill_ratio[random_id1] = sample_uniform(0., 1., (len(random_id1)), self.device)
        self.fill_ratio[soil_in] = sample_uniform(0., 1., (len(soil_in)), self.device)
        self.pre_fill_ratio[soil_in] = sample_uniform(0., 1., (len(soil_in)), self.device)
        self.pre_actions[env_ids] = torch.zeros((len(env_ids), 3), device=self.device)
        self.epi_step[env_ids] = torch.zeros(len(env_ids), device=self.device)
        
        #soil param
        self.c[env_ids] = sample_uniform(0,105000, (len(env_ids)), self.device)
        self.c_a[env_ids] = sample_uniform(0, self.c[env_ids], (len(env_ids)), self.device)
        self.sifa[env_ids] = sample_uniform(0.3, 0.8, (len(env_ids)), self.device)
        self.uw[env_ids] = sample_uniform(17000, 22000, (len(env_ids)), self.device)
        self.sbfa[env_ids] = sample_uniform(0.2, 0.4, (len(env_ids)), self.device)
        self.cpf[env_ids] = sample_uniform(0, 300, (len(env_ids)), self.device)
        self.max_L[env_ids] = torch.full_like(self.c[env_ids], self.plate_height, dtype=torch.float32)
        self.soil_mass[env_ids] = torch.full_like(self.c[env_ids], 0., dtype=torch.float32)

        self.P_cord[env_ids] = (self._robot.data.body_pos_w[env_ids, self.tip2_idx] - self.scene.env_origins[env_ids])[:,[0,2]]
        self.centroid[env_ids] =  torch.zeros((len(env_ids), 2), device=self.device)
        
        joint_ang = self._robot.data.joint_pos
        self.boom_ang = joint_ang[:,0]
        self.arm_ang = joint_ang[:,1]
        self.buck_ang = joint_ang[:,2]
        self.pre_joint_pos[env_ids] = joint_ang[env_ids].clone()
        self.joint_vel[env_ids] = torch.zeros((len(env_ids), 3), device=self.device)
        
        self.tip_pos = (self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.tip_idx] - self.scene.env_origins)[:,[0,2]]
        self.tip_pos2 = (self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.tip2_idx] - self.scene.env_origins)[:,[0,2]]
        self.tip_pos3 = (self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.tip3_idx] - self.scene.env_origins)[:,[0,2]]
        self.circle_mid = (self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.circle_mid_idx] - self.scene.env_origins)[:,[0,2]]
        
        self.tip_ang = self.batch_angle_xz(self.tip_pos2,self.tip_pos)
        self.tip_ang2 = self.batch_angle_xz(self.tip_pos3,self.tip_pos)
        self.pre_tip_ang = self.batch_angle_xz(self.tip_pos2,self.tip_pos)
        # self.tip_lin_vel = self._robot.data.body_lin_vel_w[list(range(0, self.num_envs)), self.tip_idx][:,[0,2]]
        # self.tip_ang_vel = self._robot.data.body_ang_vel_w[list(range(0, self.num_envs)), self.tip_idx][:,1]
        self.tip_lin_vel[env_ids] = torch.zeros((len(env_ids), 2), device=self.device)
        self.tip_ang_vel[env_ids] =torch.zeros((len(env_ids)), device=self.device)
        self.buck_lin_vel[env_ids] = torch.zeros((len(env_ids), 2), device=self.device)
        self.cabin_pos = (self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.cabin_idx] - self.scene.env_origins)[:,[0,2]]
        #self.ang_tip_plate[env_ids] = torch.atan2(self.tip_lin_vel[env_ids][:,1], self.tip_lin_vel[env_ids][:,0]) - self.tip_ang[env_ids]
        self.ang_tip_plate[env_ids] = torch.zeros((len(env_ids)), device=self.device)
        
        # modify_articulation_root_properties(
        #     prim_path="/World/envs/env_0/EC300E_NLC_MONO/EC300E_NLC_MONO",
        #     cfg=ArticulationRootPropertiesCfg(fix_root_link=False)
        # )
        # self.scene.reset()

    def _get_observations(self) -> dict:
        cabin_ang = self._robot.data.body_quat_w[list(range(0, self.num_envs)), self.cabin_idx]
        cabin_ang = axis_angle_from_quat(cabin_ang)[:,1]
        
        cab_denom = -self.arm_direction_angle(self.cabin_pos, self.tip_pos)
        cab_denom = torch.where(cab_denom.abs() < 1e-9, 1e-9 * (torch.sign(cab_denom) + (cab_denom == 0.).float()), cab_denom)
        cabin_ang_rate = cabin_ang / cab_denom
        
        computed_torque = self._robot.data.computed_torque
        computed_torque[:,0] = computed_torque[:,0] / self.boom_limit[1]
        computed_torque[:,1] = computed_torque[:,1] / self.arm_limit[1]
        computed_torque[:,2] = computed_torque[:,2] / self.buck_limit[1]
        
        obs = torch.cat(
            (   
                computed_torque,
                self.boom_ang.unsqueeze(-1),
                self.arm_ang.unsqueeze(-1),
                self.buck_ang.unsqueeze(-1),
                self.joint_vel,
                self.actions,
                self.soil_height.unsqueeze(-1),
                self.fill_ratio.unsqueeze(-1),
                self.tip_pos,
                self.tip_ang.unsqueeze(-1),
                self.tip_lin_vel,
                self.tip_ang_vel.unsqueeze(-1),
                cabin_ang.unsqueeze(-1),
                cabin_ang_rate.unsqueeze(-1),
                self.ang_tip_plate.unsqueeze(-1),
                torch.norm(self.tip_lin_vel, dim=-1).unsqueeze(-1),
                self.buck_lin_vel
            ),
            dim=-1,
        )
        #print(obs[0])
        #obs = torch.where(torch.isnan(obs), 0., obs)
        #print("obs", obs[obs.isnan().any(dim=-1)])
        return {"policy": obs}
        #return {"policy": torch.clamp(obs, -5.0, 5.0)}
    
    def unwrap_angle(self, angle_tensor, prev_angle_tensor):
        delta = angle_tensor - prev_angle_tensor
        delta = (delta + torch.pi) % (2 * torch.pi) - torch.pi
        return prev_angle_tensor + delta

    def batch_angle_xz(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        dx = p2[:, 0] - p1[:, 0]
        dz = p2[:, 1] - p1[:, 1]
        return torch.atan2(dz, dx)

    def compute_torque(self, desired_vel, idx):
        Kp = torch.tensor([1.0e8, 9.0e6, 6.0e5], device=self.device)
        Ki = torch.tensor([2.0e7, 8.0e7, 4.0e6], device=self.device)
        Kd = torch.tensor([0., 0., 0.], device=self.device)

        reverse_torque_scale = 0.5
        error = desired_vel - self._robot.data.joint_vel
        integral_error = self.integral_error + error * self.cfg.sim.dt
        derivative = (error - self.prev_error) / self.cfg.sim.dt
        
        if idx == 1:
            self.integral_error = integral_error
            self.prev_error = error

        # PID output
        torque = Kp * error + Ki * integral_error + Kd * derivative
       
        # Apply direction-dependent reverse torque scaling
        wrong_direction = ((desired_vel > 0) & (torque < 0)) | ((desired_vel < 0) & (torque > 0))
        torque = torch.where(wrong_direction, torque * reverse_torque_scale, torque)
        
        return torque
    
    def compute_fill_ratio(self, tip_pos):
        delta_pos = self.pre_tip_pos - tip_pos
        delta_dist = torch.norm(delta_pos, dim=-1)
        height = (self.pre_tip_pos[:,1] + tip_pos[:,1]) / 2
        depth = (self.soil_height - height).clamp(min=0.0)
        
        swept_volume = delta_dist * self.plate_width * depth
        #swept_volume = torch.where(self.ang_tip_plate < 0., 0., swept_volume)
    
        self.fill_ratio = (self.fill_ratio + swept_volume / self.bucket_vol).clamp(max=1.0)
        return
    
    def arm_direction_angle(self, p_bucket, p_base):
        delta = p_bucket - p_base  # shape (B, 2)
        theta = torch.atan2(delta[:, 1], delta[:, 0])  # z over x
        return theta  # shape (B,), radians

    
    def forward_kinematics_batch(self, theta_batch: torch.Tensor):
        # Link lengths as a tensor
        L = torch.tensor([self.boom_arm, self.arm_buck, self.buck_tip], device=theta_batch.device, dtype=theta_batch.dtype)
        
        # Initialize base positions
        x = self.boom_pos[:,0].unsqueeze(1)
        y = self.boom_pos[:,1].unsqueeze(1)
        
        x_positions = [x]
        y_positions = [y]

        for i in range(len(L)):
            # Cumulative sum of angles up to joint i
            cumulative_theta = torch.cumsum(theta_batch[:, :i+1], dim=1)

            dx = L[i] * torch.cos(cumulative_theta[:, -1])
            dy = L[i] * torch.sin(cumulative_theta[:, -1])

            new_x = x_positions[-1] - dx.unsqueeze(1)
            new_y = y_positions[-1] + dy.unsqueeze(1)

            x_positions.append(new_x)
            y_positions.append(new_y)

        # x_final = torch.cat(x_positions, dim=1)[:,3].unsqueeze(1)
        # y_final = torch.cat(y_positions, dim=1)[:,3].unsqueeze(1)

        # return torch.cat((x_final, y_final), dim=-1)
        x_final = torch.cat(x_positions, dim=1)
        y_final = torch.cat(y_positions, dim=1)

        return x_final, y_final
    
    def exc_config(self, boom_pos, arm_pos, buck_pos, tip_pos):
        boom_arm = torch.norm(boom_pos - arm_pos, 2)
        arm_buck = torch.norm(arm_pos - buck_pos, 2)
        buck_tip = torch.norm(buck_pos - tip_pos, 2)
        
        return boom_arm, arm_buck, buck_tip
    
    def soil_force(self):
        K = 1-torch.sin(self.sifa)
        bkt = self.tip_pos.clone()
        
        intersec_x, intersec_x1 = self.get_intersect()
        
        up_plate_sur = torch.cat((intersec_x1.unsqueeze(-1), self.soil_height.unsqueeze(-1)),dim=-1)
        sep_plate_sur = torch.cat((intersec_x.unsqueeze(-1), self.soil_height.unsqueeze(-1)),dim=-1)
        
        L = torch.norm(sep_plate_sur - bkt, dim=-1) # length of separation plate in contact with soil
        L = torch.where(bkt[:, 1] > self.soil_height, 0., L)
        L = torch.where(torch.isnan(intersec_x) & (bkt[:, 1] <= self.soil_height), self.max_L, L)
        L = torch.clamp(L, max=self.max_L)
        
        PO = bkt - self.P_cord
        beta = -torch.atan2(PO[:,1], PO[:,0]) # angle between ground and bucket
        beta = torch.where(beta <= 0., torch.zeros_like(beta), beta)
        
        #sep_plate_sur = torch.where(((torch.norm(sep_plate_sur - bkt, dim=-1) <= self.max_L) | (beta > 0.)).unsqueeze(-1), sep_plate_sur, self.P_cord)
        sep_plate_sur = torch.where((beta < 0.).unsqueeze(-1), self.P_cord, sep_plate_sur)
        sep_plate_sur = torch.where((self.P_cord[:,1] < sep_plate_sur[:,1]).unsqueeze(-1), self.P_cord, sep_plate_sur)
        
        lo = (torch.pi/4 - (self.sifa + self.sbfa)/2) + (torch.pi/4 - beta/2)
        
        abc_denom = torch.tan(lo)
        abc_denom = torch.where(abc_denom.abs() < 1e-9, 1e-9 * (torch.sign(abc_denom) + (abc_denom == 0.).float()), abc_denom)
        area_abc = 0.5 * L * torch.sin(beta) * (L * torch.cos(beta) + L * torch.sin(beta) / abc_denom)
        area_abx = 0.5 * torch.abs(bkt[:,0]*sep_plate_sur[:,1]+sep_plate_sur[:,0]*up_plate_sur[:,1]+up_plate_sur[:,0]*bkt[:,1] 
                                   - (bkt[:,0]*up_plate_sur[:,1]+up_plate_sur[:,0]*sep_plate_sur[:,1]+sep_plate_sur[:,0]*bkt[:,1]))
        area_abc = torch.where(bkt[:, 1] > self.soil_height, torch.zeros_like(area_abc), area_abc)
        area_abx = torch.where(bkt[:, 1] > self.soil_height, torch.zeros_like(area_abx), area_abx)
        area_abx = torch.clamp(area_abx, max = 0.423)
        area_abc = area_abc.abs()
        area_abx = area_abx.abs()
        
        area_bcx = torch.abs(area_abc - area_abx)
        
        c_s = (self.c_a * area_abx + self.c * area_bcx) / area_abc.clamp_min(1e-9)
        sifa_s = (self.sbfa * area_abx + self.sifa * area_bcx) / area_abc.clamp_min(1e-9)
        
        z_denom1 = torch.tan(lo)
        z_denom1 = torch.where(z_denom1.abs() < 1e-9, 1e-9 * (torch.sign(z_denom1) + (z_denom1 == 0.).float()), z_denom1)
        
        z_denom2 = (torch.sin(beta)*(torch.cos(beta)+torch.sin(beta)/z_denom1))
        z_denom2 = torch.where(z_denom2.abs() < 1e-9, 1e-9 * (torch.sign(z_denom2) + (z_denom2 == 0.).float()), z_denom2)
        
        z = -L/3*((torch.cos(beta)+torch.sin(beta)/torch.tan(lo))*(-torch.square(torch.sin(beta)))) / z_denom2
        ADF = self.c_a * self.B * L
        
        W_denom = torch.tan(lo)
        W_denom = torch.where(W_denom.abs() < 1e-9, 1e-9 * (torch.sign(W_denom) + (W_denom == 0.).float()), W_denom)
        W = self.uw * self.B * (0.5 * L * torch.sin(beta) * (L * torch.cos(beta) + L * torch.sin(beta) / W_denom))
        
        CF_denom = torch.sin(lo)
        CF_denom = torch.where(CF_denom.abs() < 1e-9, 1e-9 * (torch.sign(CF_denom) + (CF_denom == 0.).float()), CF_denom)
        CF = self.c * self.B * L * torch.sin(beta) / CF_denom
        
        ACF_denom = torch.tan(lo)
        ACF_denom = torch.where(ACF_denom.abs() < 1e-9, 1e-9 * (torch.sign(ACF_denom) + (ACF_denom == 0.).float()), ACF_denom)
        ACF = 0.5 * c_s * (L**2) * torch.sin(beta) * (torch.cos(beta) + torch.sin(beta)/ACF_denom)
        
        SF_denom = torch.tan(lo)
        SF_denom = torch.where(SF_denom.abs() < 1e-9, 1e-9 * (torch.sign(SF_denom) + (SF_denom == 0.).float()), SF_denom)
        SF = 0.5 * K * self.uw * z * torch.tan(sifa_s) * torch.square(L) * torch.sin(beta) * (torch.cos(beta)+ torch.sin(beta) / SF_denom)

        R_s = (-ADF * torch.cos(beta+lo+self.sifa) + W * torch.sin(lo+self.sifa) + CF * torch.cos(self.sifa) + 2 * ACF * torch.cos(self.sifa) + 2 * SF * torch.cos(self.sifa)) / torch.sin(beta+lo+self.sbfa+self.sifa).clamp_min(1e-9)
     
        #center_z = torch.norm(PO, dim=-1) / 2
        center_z = L / 2
        p_n = 0.5 * self.uw * center_z*((1+K) + (1-K)*torch.cos(2*beta))

        p = self.cpf * p_n
        R_ps = (self.c_a + p_n * torch.tan(self.sbfa)) * self.B * L
        R_pt_denom = torch.tan(torch.tensor(self.alpha, device=self.device))
        R_pt_denom = torch.where(R_pt_denom.abs() < 1e-9, 1e-9 * (torch.sign(R_pt_denom) + (R_pt_denom == 0.).float()), R_pt_denom)
        R_pt = self.n * (p + (self.c_a + p * torch.tan(self.sbfa)) * (1 / R_pt_denom)) * self.A_t
        
        R_p = R_ps + R_pt
        R_p = R_p * torch.cos(self.ang_tip_plate).clamp_min(0.)
        R_s = torch.where((bkt[:,1] > self.soil_height) | (beta <= 0.), torch.zeros_like(R_s), R_s)
        R_p = torch.where(bkt[:,1] > self.soil_height, torch.zeros_like(R_p), R_p)
        
        # O,P,E: [...,2] (월드 or 로컬)
        d = PO / (PO.norm(dim=-1, keepdim=True).clamp_min(1e-9)) # plate dir
        n = -torch.stack([-d[...,1], d[...,0]], dim=-1)         # rotate +90°

        EO = self.tip_pos2 -bkt
        d1 = EO / (EO.norm(dim=-1, keepdim=True).clamp_min(1e-9))
        
        g_world = torch.tensor([0., 0., -9.81], device=self.device, dtype=torch.float32)
        
        Fs = n * R_s.unsqueeze(-1)
        Fp = d1 * R_p.unsqueeze(-1)
        Fg = g_world * self.soil_mass[...,None] 
        
        Fs_pos = (sep_plate_sur + bkt) / 2
        #Fs_pos = torch.where(((torch.norm(sep_plate_sur - bkt, dim=-1) > L) |  torch.isnan(intersec_x)).unsqueeze(-1), (self.P_cord + bkt) / 2, Fs_pos)
        
        c, s = torch.cos(-self.sbfa), torch.sin(-self.sbfa)
        R = torch.stack((torch.stack((c, -s), -1),
                        torch.stack((s,  c), -1)), -2)          # (..., 2, 2)
        rotated_Fs = (R @ Fs.unsqueeze(-1)).squeeze(-1)     
        
        Fs_pos_y = torch.ones((self.num_envs, 1), dtype=torch.float32, device=self.device) * 0.035
        Fs_pos = torch.cat((Fs_pos[:,0].unsqueeze(-1), Fs_pos_y, Fs_pos[:,1].unsqueeze(-1)), dim=-1)[:,None,:]
        
        rotated_Fs_y = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        rotated_Fs = torch.cat((rotated_Fs[:,0].unsqueeze(-1), rotated_Fs_y, rotated_Fs[:,1].unsqueeze(-1)), dim=-1)[:,None,:]

        Fp_pos_y = torch.ones((self.num_envs, 1), dtype=torch.float32, device=self.device) * 0.035
        Fp_pos = torch.cat((self.tip_pos[:,0].unsqueeze(-1), Fp_pos_y, self.tip_pos[:,1].unsqueeze(-1)), dim=-1)[:,None,:]
    
        Fp_y = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        Fp = torch.cat((Fp[:,0].unsqueeze(-1), Fp_y, Fp[:,1].unsqueeze(-1)), dim=-1)[:,None,:]
     
        Fg_pos_y = torch.ones((self.num_envs, 1), dtype=torch.float32, device=self.device) * 0.035
        Fg_pos = torch.cat((self.centroid[:,0].unsqueeze(-1), Fg_pos_y, self.centroid[:,1].unsqueeze(-1)), dim=-1)[:,None,:]
        Fg = Fg[:,None,:]
        
        total_f = torch.cat((rotated_Fs, Fp), dim=1)
        total_pos = torch.cat((Fs_pos, Fp_pos), dim=1)
        
        Jw = self.get_jacobian()
        Fw = self.get_wrench(total_f, total_pos)
        gw = self.get_wrench(Fg, Fg_pos)
        
        tau_f = torch.einsum('bij,bj->bi', Jw.transpose(1,2), Fw)  # (B,6+n)
        tau_g = torch.einsum('bij,bj->bi', Jw.transpose(1,2), gw)
        
        #print(Fw[0][:3])
        #print(tau_gen[:, 6:][0])
        #print(tau_gen[:, 6:])

        f_tau_base   = tau_f[:, :6]   # (B,6)
        f_tau_joints = tau_f[:, 6:]   # (B,n)
        
        g_tau_base   = tau_g[:, :6]   # (B,6)
        g_tau_joints = tau_g[:, 6:]   # (B,n)
        
        # print(1, bkt[bkt.isnan().any(dim=-1)], self.P_cord[self.P_cord.isnan().any(dim=-1)])
        # print(2, self.tip_pos[self.tip_pos.isnan().any(dim=-1)], self.tip_pos2[self.tip_pos2.isnan().any(dim=-1)], self.circle_mid[self.circle_mid.isnan().any(dim=-1)])
      
        return f_tau_joints, f_tau_base, g_tau_joints, g_tau_base
    
    def get_intersect(self):
        self.solve_theta()
        eps = 1e-9

        bkt  = self.tip_pos.clone()   # (N,2): [x0, y0]
        bkt3 = self.tip_pos3.clone()  # (N,2): for gradient1

        x0 = bkt[:, 0]
        y0 = bkt[:, 1]
        dx  = self.P_cord[:, 0] - x0
        dy  = self.P_cord[:, 1] - y0
        dx1 = bkt3[:, 0] - x0
        dy1 = bkt3[:, 1] - y0

        # ----- 원래 기울기들 (수직선 마스크 처리) -----
        vert0 = dx.abs()  < eps
        vert1 = dx1.abs() < eps

        # 안전분모로 임시계산 후, 수직은 inf로 덮어쓰기
        gradient  = dy  / torch.where(vert0, torch.ones_like(dx),  dx)
        gradient  = torch.where(vert0, torch.full_like(gradient,  float('inf')), gradient)

        gradient1 = dy1 / torch.where(vert1, torch.ones_like(dx1), dx1)
        gradient1 = torch.where(vert1, torch.full_like(gradient1, float('inf')), gradient1)

        # ----- y = soil_height 와의 교점 x -----
        dh = self.soil_height - y0  # (N,)

        # 회전선과의 교점
        hor_rot = gradient.abs() == 0.  # 수평
        intersec_x = torch.where(
            vert0,                          # 수직선이면 x = x0
            x0,
            torch.where(
                hor_rot,                       # 수평선이면
                torch.where(dh.abs() == 0.,    # 같은 높이면 x0, 아니면 교점 없음(NaN)
                            x0,
                            torch.full_like(x0, float('nan'))),
                (dh / gradient) + x0       # 일반 케이스
            )
        )

        # 원래 gradient1 선과의 교점
        hor1 = gradient1.abs() == 0.
        intersec_x1 = torch.where(
            vert1,
            x0,
            torch.where(
                hor1,
                torch.where(dh.abs() == 0., x0, torch.full_like(x0, float('nan'))),
                (dh / gradient1) + x0
            )
        )
        
        return intersec_x, intersec_x1
    
    def cross2(self, a, b):
        # a,b: [...,2]
        return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]
    
    def ensure_B(self, x, B, device, dtype):
        """x -> [B] 텐서로 강제"""
        t = torch.as_tensor(x, device=device, dtype=dtype)
        if t.ndim == 0:             # scalar(float 등)
            t = t.expand(B)         # [B] (view)
        elif t.ndim == 1 and t.shape[0] == B:
            pass
        else:
            raise ValueError(f"Expected scalar or [B], got {tuple(t.shape)}")
        return t

    def ensure_B2(self, x, B, device, dtype):
        """x -> [B,2] 텐서로 강제"""
        t = torch.as_tensor(x, device=device, dtype=dtype)
        if t.ndim == 1 and t.shape[0] == 2:        # [2]
            t = t.unsqueeze(0).expand(B, 2)        # [B,2]
        elif t.ndim == 2 and t.shape[0] == B and t.shape[1] == 2:
            pass
        else:
            raise ValueError(f"Expected [2] or [B,2], got {tuple(t.shape)}")
        return t

    def A_of_theta_world(self, theta, theta_E):
        """
        theta: [B] 또는 [B,K]
        반환: A(theta) [B] 또는 [B,K], P(theta) [...,2]
        """
        device, dtype = theta.device, theta.dtype
        B = theta.shape[0]

        # --- 월드 파라미터를 [B]/[B,2]로 강제 ---
        R = self.ensure_B(self.bucket_circle, B, device, dtype)        # [B]
        O = self.ensure_B2(self.tip_pos,        B, device, dtype)      # [B,2]
        E = self.ensure_B2(self.tip_pos2,       B, device, dtype)      # [B,2]
        C = self.ensure_B2(self.circle_mid,     B, device, dtype)      # [B,2]

        if theta.dim() == 1:  # [B]
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)           # [B]
            P = C + torch.stack([R * cos_t, R * sin_t], dim=-1)         # [B,2]
            EO = E - O                                                  # [B,2]
            PO = P - O                                                  # [B,2]
            cross_raw = self.cross2(EO, PO)                                 # [B]
            Atr = 0.5 * cross_raw.abs()

            dtheta = torch.clamp(theta_E - theta, min=0.0)              # [B]
            Aseg = 0.5 * (R**2) * (dtheta - torch.sin(dtheta))          # [B]
            return Atr, Aseg

        else:  # [B,K]
            B, K = theta.shape
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)           # [B,K]
            # 브로드캐스팅 정렬
            P  = C[:, None, :] + torch.stack([R[:, None]*cos_t,
                                              R[:, None]*sin_t], dim=-1)  # [B,K,2]
            EO = (E - O)[:, None, :]                                      # [B,1,2]
            PO = P - O[:, None, :]                                        # [B,K,2]
            cross_raw = self.cross2(EO, PO)                                   # [B,K]
            Atr = 0.5 * cross_raw.abs()

            dtheta = torch.clamp(theta_E[:, None] - theta, min=0.0)       # [B,K]
            Aseg = 0.5 * (R[:, None]**2) * (dtheta - torch.sin(dtheta))   # [B,K]
            
            return Atr, Aseg                                             # [B,K], [B,K,2]

    def dA_dtheta_world(self, theta):
        """theta: [B]"""
        device, dtype = theta.device, theta.dtype
        B = theta.shape[0]
        R = self.ensure_B(self.bucket_circle, B, device, dtype)
        O = self.ensure_B2(self.tip_pos,    B, device, dtype)
        E = self.ensure_B2(self.tip_pos2,   B, device, dtype)
        C = self.ensure_B2(self.circle_mid, B, device, dtype)

        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        P   = C + torch.stack([R * cos_t, R * sin_t], dim=-1)                 # [B,2]
        P1  = torch.stack([R * sin_t, -R * cos_t], dim=-1)                    # [B,2]

        EO = E - O
        cross_raw = self.cross2(EO, P - O)                                        # [B]
        sgn = torch.where(torch.sign(cross_raw) == 0,
                          torch.ones_like(cross_raw), torch.sign(cross_raw))
        tri_term = 0.5 * sgn * self.cross2(EO, P1)                                # [B]

        EC = E - C
        theta_E = torch.atan2(EC[..., 1], EC[..., 0])                         # [B]
        dtheta = theta_E - theta
        seg_term = 0.5 * (R**2) * (1.0 - torch.cos(torch.clamp(dtheta, min=0.0)))
        seg_term = torch.where(dtheta >= 0.0, seg_term, torch.zeros_like(seg_term))
        return tri_term + seg_term

    @torch.no_grad()
    def solve_theta(self, K: int = 512, eps: float = 1e-12):
        """테이블 룩업 → 선형보간 → 뉴턴 1회 (월드)"""
        # θ 그리드
        EC = self.tip_pos2 - self.circle_mid
        theta_E   = torch.atan2(EC[..., 1], EC[..., 0])    # [B]

        w = torch.linspace(0.0, 1.0, K, device=theta_E.device, dtype=theta_E.dtype)
        theta_grid = theta_E[:, None] - torch.pi * w[None, :]  # [B,K]
        Atr_grid, Aseg_grid = self.A_of_theta_world(theta_grid, theta_E)
        A_grid = Atr_grid + Aseg_grid
        A_grid[:,0] = 0.
        
        # 타깃 면적 클램프
        #A_tgt = self.fill_ratio
        A_tgt = self.fill_ratio* (0.5 * torch.pi * self.bucket_circle**2 + self.plate_height * self.bucket_circle)

        # 구간 인덱스 + 보간
        idx = torch.sum(A_grid < A_tgt[:, None], dim=1, keepdim=True).clamp(1, K-1)
        i0, i1 = idx - 1, idx
        A0, A1   = A_grid.gather(1, i0), A_grid.gather(1, i1)
        th0, th1 = theta_grid.gather(1, i0), theta_grid.gather(1, i1)
        t = (A_tgt[:, None] - A0) / (A1 - A0)
        theta = (th0 + t * (th1 - th0)).squeeze(1)        # [B]

        # # 뉴턴 1회
        # f  = self.A_of_theta_world(theta0, theta_E)[0] - A_tgt
        # df = self.dA_dtheta_world(theta0)
        # theta = torch.clamp(theta0 - f/df, theta_E, theta_max)
        A_tri, A_seg = self.A_of_theta_world(theta, theta_E)
        A_tot = A_tri + A_seg
        self.soil_mass = (A_tot * self.plate_width * self.uw).clamp_min(0.0) 
        self.soil_mass = torch.where(self.fill_ratio==0., 0., self.soil_mass)
        
        P_theta = torch.where(theta < -torch.pi, 2*torch.pi + theta, theta)

        # 최종 P
        R = self.ensure_B(self.bucket_circle, P_theta.shape[0], P_theta.device, P_theta.dtype)
        C = self.ensure_B2(self.circle_mid, P_theta.shape[0], P_theta.device, P_theta.dtype)
        P = C + torch.stack([R * torch.cos(P_theta), R * torch.sin(P_theta)], dim=-1)
        
        # print("theta",theta[P.isnan().any(dim=-1)])
        # print(theta)
        
        self.P_cord = P
        self.max_L  = torch.norm(P, dim=-1)
        
        c_tri_2d = (self.tip_pos + self.tip_pos2 + P) / 3.0 
        dth = theta_E -theta
        theta_m = theta_E - 0.5*dth
        s = torch.sin(0.5*dth)
        denom = (dth - torch.sin(dth)).clamp_min(1e-9)
        r_seg = (4.0*self.bucket_circle*s*s*s) / (3.0*denom)                            # [B]
        c_seg_2d = torch.stack([r_seg*torch.cos(theta_m), r_seg*torch.sin(theta_m)], dim=-1) + C    # [B,2]
        self.centroid = (A_tri.unsqueeze(-1)*c_tri_2d + A_seg.unsqueeze(-1)*c_seg_2d) / A_tot.unsqueeze(-1)
        
        return


    def skew(self, v: torch.Tensor) -> torch.Tensor:
        """v: (...,3) -> [v]_x: (...,3,3)"""
        x, y, z = v[..., 0], v[..., 1], v[..., 2]
        O = torch.zeros((*v.shape[:-1], 3, 3), dtype=v.dtype, device=v.device)
        O[..., 0, 1] = -z; O[..., 0, 2] =  y
        O[..., 1, 0] =  z; O[..., 1, 2] = -x
        O[..., 2, 0] = -y; O[..., 2, 1] =  x
        return O

    def Xtrans_twist(self, p: torch.Tensor) -> torch.Tensor:
        """
        Twist 기준점 이동: o -> o+p (회전 없음).
        p: (B,3)
        return: (B,6,6)
        """
        B = p.shape[0]
        I6 = torch.eye(6, dtype=p.dtype, device=p.device).expand(B, 6, 6).clone()
        I6[:, :3, 3:] = -self.skew(p)
        return I6

    # -------------------------
    # Jacobian (batch)
    # -------------------------
    def get_jacobian(self) -> torch.Tensor:
        """
        Returns:
        J_w: (B, 6, 6+n)   월드 프레임, 기준점 p_w
        """
        base_origin_w = self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.base_idx] - self.scene.env_origins
        
        joint_axes_w = torch.tensor([[[0.0, 1.0, 0.0],
                                    [0.0,-1.0, 0.0],
                                    [0.0,-1.0, 0.0]]] * self.num_envs, dtype=torch.float32, device=self.device)  # (B,n,3)
        
        
        idxs = [self.boom_idx, self.arm_idx, self.buck_idx]
        joint_origins_w = self._robot.data.body_pos_w[torch.arange(self.num_envs).unsqueeze(-1), idxs] - self.scene.env_origins.unsqueeze(1)
        
        p_w = joint_origins_w[:, -1, :]  # (B,3)
        
        B, n, _ = joint_origins_w.shape
        J = torch.zeros((B, 6, 6+n), dtype=p_w.dtype, device=p_w.device)

        # base 6DoF block
        Xb = self.Xtrans_twist(p_w - base_origin_w)  # (B,6,6)
        J[:, :, :6] = Xb

        # revolute columns
        r = p_w[:, None, :] - joint_origins_w   # (B,n,3)
        v = torch.cross(joint_axes_w, r, dim=-1) # (B,n,3)
        w = joint_axes_w                         # (B,n,3)

        J[:, :3, 6:] = v.transpose(1, 2)  # (B,3,n) -> (B,3,n)
        J[:, 3:, 6:] = w.transpose(1, 2)  # (B,3,n)
        return J


    # -------------------------
    # Wrench from multiple contact points (batch)
    # -------------------------
    def get_wrench(
        self,
        f_list: torch.Tensor,       # (B,K,3)  forces
        p_app_list: torch.Tensor,   # (B,K,3)  application points
    ) -> torch.Tensor:
        """
        Return: F_w: (B,6)
        """
        p_ref_w = self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.buck_idx] - self.scene.env_origins
        f_sum = f_list.sum(dim=1)  # (B,3)
        m_terms = torch.cross(p_app_list - p_ref_w[:, None, :], f_list, dim=-1)  # (B,K,3)
        m_sum = m_terms.sum(dim=1)  # (B,3)
        return torch.cat([f_sum, m_sum], dim=-1)  # (B,6)
