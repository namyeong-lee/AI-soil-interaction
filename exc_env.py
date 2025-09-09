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

@configclass
class ExcEnvCfg(DirectRLEnvCfg):
    # env
    n_steps=681
    episode_length_s = 19.95  # 500 timesteps
    decimation = 18
    action_space = 3
    observation_space = 23
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/120,
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
        collision_group = -1,
        prim_path="/World/envs/env_.*/EC300E_NLC_MONO",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/volvo/Downloads/exc.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
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
                effort_limit_sim=100.0,
                velocity_limit_sim=100.0,
                stiffness=10000.0,
                damping=100.0,
            ),
            "arm_act": ImplicitActuatorCfg(
                joint_names_expr=["arm_joint"],
                effort_limit_sim=100.0,
                velocity_limit_sim=100.0,
                stiffness=10000.0,
                damping=100.0,
            ),
            "bucket_act": ImplicitActuatorCfg(
                joint_names_expr=["buck_joint"],
                effort_limit_sim=100.0,
                velocity_limit_sim=100.0,
                stiffness=10000.0,
                damping=100.0,
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
        self.actions = torch.zeros((self.num_envs, 3), device=self.device)
        self.torque = torch.zeros((self.num_envs, 3), device=self.device)
        self.v_max = torch.tensor([2,2,2], device=self.device)
        self.v_min = torch.tensor([2,2,2], device=self.device)
        self.plate_width, self.plate_height = 1.2, 0.94
        self.bucket_circle = 0.45
        self.bucket_vol = (self.bucket_circle * self.plate_height * self.plate_width) + (0.5 * torch.pi * (self.bucket_circle**2) * self.plate_width)
        self.pre_tip_pos = (self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.tip_idx] - self.scene.env_origins)[:,[0,2]]
        self.total_swept = torch.zeros(self.num_envs, device=self.device)
        self.fill_ratio = torch.zeros(self.num_envs, device=self.device)
        self.epi_step = torch.zeros(self.num_envs, device=self.device)
        self.max_step = self.cfg.episode_length_s / self.dt
        self.pre_fill_ratio = torch.zeros(self.num_envs, device=self.device)
        self.pre_actions = torch.zeros((self.num_envs, 3), device=self.device)
        self.count_steps, self.tip_pos, self.tip_lin_vel, self.cabin_pos, self.base_pos, self.tip_pos2, self.tip_pos3, self.tip_ang, self.circle_mid  = 0., 0.,0.,0.,0.,0.,0., 0., 0.
        self.tip_ang_vel, self.ang_tip_plate, self.cabin_lin_vel, self.boom_ang, self.arm_ang, self.buck_ang = 0.,0.,0.,0.,0.,0.
        self.kj, self.nc1, self.nc2, self.nc3, self.nc4, self.nc5, self.nc6 = 0.,0.,0.,0.,0.,0.,0.
        self.fill_terminal_reward, self.close_terminal_reward = 0., 0.
        
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
        self.c = sample_uniform(0,105, (self.num_envs), self.device)
        self.c_a = sample_uniform(0, self.c, (self.num_envs), self.device)
        self.sifa = sample_uniform(0.3, 0.8, (self.num_envs), self.device)
        self.uw = sample_uniform(17, 22, (self.num_envs), self.device)
        self.sbfa = sample_uniform(0.2, 0.4, (self.num_envs), self.device)
        self.cpf = sample_uniform(0, 300, (self.num_envs), self.device)
        
        self.A_s = self.plate_width * self.plate_height # area of separation plate
        self.n = 5 # number of teeth
        self.alpha = 0.5 # tip semi angle
        self.tooth_r = 0.075 # tooth radius
        self.A_t = torch.pi * self.tooth_r**2
        self.B = self.plate_width # separation plate width
        self.max_L = torch.full_like(self.c, self.plate_height, dtype=torch.float32)
        self.P_cord = torch.zeros((self.num_envs, 2), device=self.device)
        
        self.tau_base   = 0.   # (B,6)
        self.tau_joints = 0.   # (B,n)
        
        self.writer = SummaryWriter(f"/home/volvo/soil_vs/runs")
        
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
        self.actions =  actions.clone()
        # self.actions = 0.5 * (self.actions + 1.0) * (self.v_max - self.v_min) + self.v_min
        # self.actions = torch.full_like(self._robot.data.joint_vel, 0.5, dtype=torch.float32)
        # self.actions[:, 0] = torch.ones(self.num_envs) *0.3
        # self.actions[:, 1] = torch.ones(self.num_envs) *-0.5
        # self.actions[:, 2] = torch.ones(self.num_envs) *-0.6
        
        self.actions[:,0] = torch.clamp(self.actions[:,0], -0.3, 0.3)
        self.actions[:,1] = torch.clamp(self.actions[:,1], -0.5, 0.5)
        self.actions[:,2] = torch.clamp(self.actions[:,2], -0.6, 0.6)
        
        self.torque = self.compute_torque(self.actions)
        self.torque = self.torque + self.tau_joints
        
        torque_limit = torch.tensor([-3e5, 3e5], device=self.device)
        self.torque = torch.clamp(self.torque, torque_limit[0], torque_limit[1])
        
    # post-physics step calls    
    def _apply_action(self):
        #body_ids = torch.full_like(torch.arange(self.num_envs), self.base_idx, dtype=torch.int32, device=self.device)

        self.writer.add_scalar("tau_base", torch.max(self.tau_base), self.count_steps)
        
        #self._robot.set_external_force_and_torque(self.tau_base[:,:3][:,None,:], self.tau_base[:,3:][:,None,:], body_ids=self.base_idx)
        
        #self.torque[:,[1,2]] = torch.zeros((self.num_envs,2), device=self.device)
        # self.torque[:,0] = torch.ones(self.num_envs) * -10000000.
        # self.torque[:,1] = torch.ones(self.num_envs) * -1000000.
        # self.torque[:,2] = torch.ones(self.num_envs) * -100000.
        self._robot.set_joint_effort_target(self.torque)
        #self._robot.set_joint_effort_target(torch.tensor([[-1000.,1000,1000],[1000.,-1000,-1000],[1000.,-1000,-1000],[1000.,-1000,-1000],[-1000.,1000,1000]]))
        #self._robot.set_joint_effort_target(torch.tensor([[-1000.,-1000,-1000]]))
        
        #self._robot.set_joint_effort_target(torch.tensor([[150000.,-1050000.,-1050000.],[1500000.,-1050000.,-1050000.],[1500000.,1050000.,-1050000.]]))
        
        # rb_f = torch.zeros((self.num_envs, 1, 3), device=self.device)
        # rb_tau = torch.zeros_like(rb_f)
        # #rb_f[:,0] = torch.tensor([0., 0., -470000.], device=self.device)
        # rb_f[:,0] = torch.tensor([-3000000., 0., 2500000.], device=self.device)
        # self._robot.set_external_force_and_torque(rb_f, rb_tau, body_ids=self.buck_idx)
        
        # rb_f = torch.zeros((self.num_envs, 1, 3), device=self.device)
        # rb_tau = torch.zeros_like(rb_f)
        # rb_f[:,0] = torch.tensor([2500000., 0., 0.], device=self.device)
        # rb_f1 = torch.zeros((self.num_envs, 1, 3), device=self.device)
        # rb_tau1 = torch.zeros_like(rb_f1)
        # rb_f1[:,0] = torch.tensor([0., 0., -2500000], device=self.device)
        # pos = torch.zeros((self.num_envs, 1, 3), device=self.device)
        # pos[:,0] = torch.tensor([10., 0., 0.], device=self.device)
        # pos1 = torch.zeros((self.num_envs, 1, 3), device=self.device)
        # pos1[:,0] = torch.tensor([0., 0., 0.], device=self.device)
        # self._robot.set_external_force_and_torque(rb_f, rb_tau, positions=pos, body_ids=self.buck_idx)
        
        # print(rb_f1.shape, rb_tau1.shape, self.tip_idx)
        # self._robot.set_external_force_and_torque(rb_f1, rb_tau1, positions=pos1, body_ids=self.tip_idx)
        return

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cabin_lin_vel = self._robot.data.body_lin_vel_w[list(range(0, self.num_envs)), self.cabin_idx][:,[0,2]]
        buck_pos = (self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.buck_idx] - self.scene.env_origins)[:,[0,2]]
        j = self.count_steps // self.cfg.n_steps
        self.kj = 1 - torch.exp(torch.tensor(-0.01 * j))
        self.count_steps += 1
        self.epi_step += 1
        
        self.nc1 = torch.norm(self.tip_lin_vel, dim=-1) > 0.5
        self.nc2 = self.ang_tip_plate < 0.
        self.nc3 = torch.norm(cabin_lin_vel, dim=-1) > 0.1
        self.nc4 = (self.fill_ratio == 0.) & ((self.tip_pos[:,1] - self.soil_height) > 0.5)
        self.nc5 = ((self.tip_pos[:,0] > -2.5) & ((self.cabin_pos[:,1] + 2.) > self.tip_pos[:,1])) | (self.boom_ang <= self.joint_lower_limits[0]) | (self.boom_ang >= self.joint_upper_limits[0]) | (self.arm_ang <= self.joint_lower_limits[1]) | (self.arm_ang >= self.joint_upper_limits[1]) | (self.buck_ang <= self.joint_lower_limits[2]) | (self.buck_ang >= self.joint_upper_limits[2])
        self.nc6 = (self.fill_ratio == 1.) & ((self.P_cord[:,1] <  self.soil_height) | (self.tip_pos[:,1] < self.soil_height)) 
        
        pc1 = self.fill_ratio > self.kj * 0.68 + 0.3
        pc2 = torch.norm(self.cabin_pos - self.tip_pos, dim=-1) < 3.0
        pc3 = (self.tip_pos[:,1] - self.soil_height) > self.kj * 0.7 + 0.3
        pc4 = (self.tip_ang + 1.0388) < 0.3
        pc5 = buck_pos[:,1] > self.tip_pos[:,1]
        
        positive_terminations = torch.where((pc1 | pc2) & pc3 & pc4 & pc5, 1., 0.)
        negative_terminations = torch.where(self.nc1 | self.nc2 | self.nc3 | self.nc4 | self.nc5 | self.nc6, 1., 0.)
        
        self.fill_terminal_reward = torch.where(pc1 & pc3 & pc4 & pc5, 10., 0.)
        self.close_terminal_reward = torch.where(pc2 & pc3 & pc4 & pc5, 5., 0.)
        
        positive_terminations = torch.where(self.max_step == self.epi_step, 1., positive_terminations)
        
        # if self.count_steps % 1000 ==0:
        #     a=1.
        #     
        # else:
        #     a=0.    
        
        # return torch.tensor([a]*self.num_envs), torch.tensor([a]*self.num_envs)
        return positive_terminations, negative_terminations

    def _get_rewards(self) -> torch.Tensor:
        
        c1 = self.fill_ratio > (self.kj * 0.6 + 0.3)
        c2 = torch.norm(self.cabin_pos - self.tip_pos, dim=-1) < 3.0
        c3 = (self.tip_pos[:,1] - self.soil_height) > 1.0
        c4 = (self.tip_ang + 1.0388) < 0.3
        c5 = torch.norm(self.tip_lin_vel, dim=-1) < 0.4
        
        move_down = torch.where((self.fill_ratio <0.05) & c5, -0.1*self.tip_lin_vel[:,1], 0.)
           
        filling = torch.where(~c2 & c5, self.fill_ratio - self.pre_fill_ratio, 0.)
        self.pre_fill_ratio = self.fill_ratio
        
        move_up = torch.where(((c1 | c2) & ~c3) & c5, 0.1*self.tip_lin_vel[:,1], 0.)
        
        curl = torch.where(((c1 | c2) & ~c4) & c5, 0.05*self.tip_ang_vel, 0.)
          
        smooth = -0.005 * torch.norm(self.actions - self.pre_actions, p=1, dim=-1)
        self.pre_actions = self.actions
        
        reward = move_down + filling + move_up + curl + smooth
        reward = torch.where(self.nc1 | self.nc2 | self.nc3 | self.nc4 | self.nc5 | self.nc6, -1., reward)
        
        total_reward = reward + self.fill_terminal_reward + self.close_terminal_reward
        
        return total_reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        
        # cab_ang = sample_uniform(0., 0., len(env_ids), self.device)
        # ang = quat_from_angle_axis(cab_ang, torch.tensor([0.0, 1.0, 0.0], device=self.device))
        # self._robot.write_root_pose_to_sim(torch.cat((self.fixed_pos, ang), dim=-1), env_ids=env_ids)
        # self.object.write_root_pose_to_sim(torch.cat((self.obj_pos, ang), dim=-1), env_ids=env_ids)
        
        # #boom_ang = sample_uniform(-0.5, -0.5, (len(env_ids), 1), self.device)
        # #arm_ang = sample_uniform(-0.1, -0.1, (len(env_ids), 1), self.device)
        # buck_ang = sample_uniform(-0.1, -0.1, (len(env_ids), 1), self.device)
        # boom_ang = sample_uniform(-1.56, -1.56, (len(env_ids), 1), self.device)
        # arm_ang = sample_uniform(-2.08, -2.08, (len(env_ids), 1), self.device)
        # #buck_ang = sample_uniform(-2.96, -2.96, (len(env_ids), 1), self.device)
        # joint_pos = torch.cat((boom_ang, arm_ang, buck_ang),-1)
        # joint_vel = torch.zeros_like(joint_pos)
        # self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        # self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        # self.scene.update(dt=self.physics_dt)
        
        # robot state
        cab_ang = sample_uniform(-0.1, 0.1, len(env_ids), self.device)
        ang = quat_from_angle_axis(cab_ang, torch.tensor([0.0, 1.0, 0.0], device=self.device))
        
        self._robot.write_root_pose_to_sim(torch.cat((self.fixed_pos[env_ids], ang), dim=-1), env_ids=env_ids)
        self.object.write_root_pose_to_sim(torch.cat((self.obj_pos[env_ids], ang), dim=-1), env_ids=env_ids)
        
        copy_env_ids = env_ids.clone()
        while True:
            boom_ang = sample_uniform(self.joint_lower_limits[0]+0.1, self.joint_upper_limits[0]-0.1, (len(copy_env_ids), 1), self.device)
            arm_ang = sample_uniform(self.joint_lower_limits[1]+0.1, self.joint_upper_limits[1]-0.1, (len(copy_env_ids), 1), self.device)
            buck_ang = sample_uniform(self.joint_lower_limits[2]+0.1, self.joint_upper_limits[2]-0.1, (len(copy_env_ids), 1), self.device)
            joint_pos = torch.cat((boom_ang, arm_ang, buck_ang),-1)
            joint_vel = torch.zeros_like(joint_pos)
            self._robot.set_joint_position_target(joint_pos, env_ids=copy_env_ids)
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=copy_env_ids)
            self.scene.update(dt=self.physics_dt)
            
            self.tip_pos = (self._robot.data.body_pos_w[copy_env_ids, self.tip_idx] - self.scene.env_origins[copy_env_ids])[:,[0,2]]
            
            condition = (self.tip_pos[:, 1] - self.soil_height[copy_env_ids]) >= 0.4
            copy_env_ids = copy_env_ids[condition]
            
            if copy_env_ids.numel() ==0:
                break
            
        random = sample_uniform(0, 1, (len(env_ids)), self.device)
        random_id = torch.where(random<0.25)[0]
        random_id1 = torch.where(random<0.25)[0]
        
        while True:
            boom_ang = sample_uniform(self.joint_lower_limits[0]+0.1, self.joint_upper_limits[0]-0.1, (len(random_id), 1), self.device)
            arm_ang = sample_uniform(self.joint_lower_limits[1]+0.1, self.joint_upper_limits[1]-0.1, (len(random_id), 1), self.device)
            buck_ang = sample_uniform(self.joint_lower_limits[2]+0.1, self.joint_upper_limits[2]-0.1, (len(random_id), 1), self.device)
            joint_pos = torch.cat((boom_ang, arm_ang, buck_ang),-1)
            joint_vel = torch.zeros_like(joint_pos)
            self._robot.set_joint_position_target(joint_pos, env_ids=random_id)
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=random_id)
            self.scene.update(dt=self.physics_dt)
            
            self.tip_pos = (self._robot.data.body_pos_w[random_id, self.tip_idx] - self.scene.env_origins[random_id])[:,[0,2]]

            condition = (self.tip_pos[:, 1] - self.soil_height[random_id]) >= 0.
            random_id = random_id[condition]

            if random_id.numel() ==0:
                break
        
        self.soil_height[env_ids] = sample_uniform(-2.0, -0.5, (len(env_ids)), self.device)
        self.prev_error[env_ids] = torch.zeros((len(env_ids), 3), device=self.device)
        self.integral_error[env_ids] = torch.zeros((len(env_ids), 3), device=self.device)
        self.actions[env_ids] = torch.zeros((len(env_ids), 3), device=self.device)
        self.pre_tip_pos[env_ids] = (self._robot.data.body_pos_w[env_ids, self.tip_idx] - self.scene.env_origins[env_ids])[:,[0,2]]
        self.total_swept[env_ids] = torch.zeros(len(env_ids), device=self.device)
        self.fill_ratio[env_ids] = torch.zeros(len(env_ids), device=self.device)
        self.pre_fill_ratio[env_ids] = torch.zeros(len(env_ids), device=self.device)
        self.fill_ratio[random_id1] = sample_uniform(0, 1, (len(random_id1)), self.device)
        self.pre_fill_ratio[random_id1] = sample_uniform(0, 1, (len(random_id1)), self.device)
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
        self.P_cord[env_ids] = torch.zeros_like(self.pre_tip_pos[env_ids])
        

    def _get_observations(self) -> dict:
        joint_ang = self._robot.data.joint_pos
        self.boom_ang = joint_ang[:,0]
        self.arm_ang = joint_ang[:,1]
        self.buck_ang = joint_ang[:,2]
        
        self.tip_pos = (self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.tip_idx] - self.scene.env_origins)[:,[0,2]]
        self.tip_pos2 = (self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.tip2_idx] - self.scene.env_origins)[:,[0,2]]
        self.tip_pos3 = (self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.tip3_idx] - self.scene.env_origins)[:,[0,2]]
        self.circle_mid = (self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.circle_mid_idx] - self.scene.env_origins)[:,[0,2]]
        
        
        self.tip_ang = -self.batch_angle_xz(self.tip_pos2,self.tip_pos)
        self.tip_lin_vel = self._robot.data.body_lin_vel_w[list(range(0, self.num_envs)), self.tip_idx][:,[0,2]]
        self.tip_ang_vel = self._robot.data.body_ang_vel_w[list(range(0, self.num_envs)), self.tip_idx][:,1]
  
        cabin_ang = self._robot.data.body_quat_w[list(range(0, self.num_envs)), self.cabin_idx]
        cabin_ang = axis_angle_from_quat(cabin_ang)[:,1]
        self.cabin_pos = (self._robot.data.body_pos_w[list(range(0, self.num_envs)), self.cabin_idx] - self.scene.env_origins)[:,[0,2]]
        
        
        cab_denom = -self.arm_direction_angle(self.cabin_pos, self.tip_pos)
        cab_denom = torch.where(cab_denom.abs() < 1e-9, 1e-9 * torch.sign(cab_denom), cab_denom)
        cabin_ang_rate = cabin_ang / cab_denom
        
        self.ang_tip_plate = self.tip_ang+torch.atan2(self.tip_lin_vel[:,1], self.tip_lin_vel[:,0])
        
        print(self.tip_lin_vel)
        
        self.fill_ratio = self.compute_fill_ratio(self.tip_pos)
        self.pre_tip_pos = self.tip_pos
        
        self.soil_force()
        
        obs = torch.cat(
            (   
                self._robot.data.computed_torque/3e5,
                joint_ang,
                self._robot.data.joint_vel,
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
            ),
            dim=-1,
        )
        #obs = torch.where(torch.isnan(obs), 0., obs)
        
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

    def compute_torque(self, desired_vel):
        # Kp = torch.tensor([16000000., 16000000., 7.0e5], device=self.device)
        # Ki = torch.tensor([8000000., 8000000., 2.5e6], device=self.device)
        # Kd = torch.tensor([0., 0.,1700.], device=self.device)
        # Kp = torch.tensor([2.5e7, 3.2e6, 1.9e5], device=self.device)
        # Ki = torch.tensor([6.0e7, 3.7e7, 2.8e6], device=self.device)
        # Kd = torch.tensor([0., 2000., 1000.], device=self.device)
        Kp = torch.tensor([1.1e7, 5.5e5, 8.9e4], device=self.device)
        Ki = torch.tensor([1.4e6, 1.4e7, 6.3e5], device=self.device)
        Kd = torch.tensor([200., 0., 0.], device=self.device)
        torque_limit = torch.tensor([-3e5, 3e5], device=self.device)
        reverse_torque_scale = 0.5
        
        #print(self._robot.data.joint_vel[0][2])
        # self.writer.add_scalar("boom/velocity", self._robot.data.joint_vel[0][0], self.count_steps)
        # self.writer.add_scalar("arm/velocity", self._robot.data.joint_vel[0][1], self.count_steps)
        # self.writer.add_scalar("bucket/velocity", self._robot.data.joint_vel[0][2], self.count_steps)
        
        # self.writer.add_scalar("boom/velocity", self.torque[0][0], self.count_steps)
        # self.writer.add_scalar("arm/velocity", self.torque[0][1], self.count_steps)
        # self.writer.add_scalar("bucket/velocity", self.torque[0][2], self.count_steps)
        
        error = desired_vel - self._robot.data.joint_vel
        self.integral_error = self.integral_error + error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        # PID output
        torque = Kp * error + Ki * self.integral_error + Kd * derivative
       
        # Apply direction-dependent reverse torque scaling
        wrong_direction = ((desired_vel > 0) & (torque < 0)) | ((desired_vel < 0) & (torque > 0))
        torque = torch.where(wrong_direction, torque * reverse_torque_scale, torque)

        # Clamp to torque limits
        torque = torch.clamp(torque, torque_limit[0], torque_limit[1])
        
        return torque
    
    def compute_fill_ratio(self, tip_pos):
        delta_pos = self.pre_tip_pos - tip_pos
        delta_dist = torch.norm(delta_pos, dim=-1)
        height = (self.pre_tip_pos[:,1] + tip_pos[:,1]) / 2
        depth = (self.soil_height - height).clamp(min=0.0)
        
        swept_volume = delta_dist * self.plate_width * depth
    
        self.total_swept = self.total_swept + swept_volume
        
        return (self.total_swept / self.bucket_vol).clamp(max=1.0)
    
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
        bkt = self.tip_pos
        # bkt2 = self.tip_pos2
        # bkt3 = self.tip_pos3
        # dx = bkt2[:, 0]-bkt[:, 0]
        # dx1 = bkt3[:, 0]-bkt[:, 0]
        
        # dx = torch.where(dx.abs()<1e-9, 1e-9, dx)
        # dx1 = torch.where(dx1.abs()<1e-9, 1e-9, dx1)
        
        # gradient = (bkt2[:, 1]-bkt[:, 1]) / dx
        # gradient1 = (bkt3[:, 1]-bkt[:, 1]) / dx1
        
        # dx2 = 1 - gradient * torch.tan(self.solve_theta)
        # dx2 = torch.where(dx2.abs()<1e-9, 1e-9, dx2)
        # new_gradient = (gradient + torch.tan(self.solve_theta)) / dx2
        
        # intersec_x = ((self.soil_height-bkt[:, 1]) / new_gradient) + bkt[:, 0]
        # intersec_x1 = ((self.soil_height-bkt[:, 1]) / gradient1) + bkt[:, 0]
        
        intersec_x, intersec_x1 = self.get_intersect()
        
        up_plate_sur = torch.cat((intersec_x1.unsqueeze(-1), self.soil_height.unsqueeze(-1)),dim=-1)
        sep_plate_sur = torch.cat((intersec_x.unsqueeze(-1), self.soil_height.unsqueeze(-1)),dim=-1)
        
        L = torch.norm(sep_plate_sur - bkt, dim=-1) # length of separation plate in contact with soil
        L = torch.where(bkt[:, 1] > self.soil_height, 0., L)
        L = torch.where(torch.isnan(intersec_x) & (bkt[:, 1] <= self.soil_height), self.max_L, L)
        L = torch.clamp(L, max=self.max_L)
        
        PO = bkt - self.P_cord
        beta = -torch.atan2(PO[:,1], PO[:,0]) # angle between ground and bucket
        #beta = -torch.atan2(b[:,1],b[:,0]) 
        beta = torch.where(torch.isnan(intersec_x), torch.zeros_like(beta), beta)
        
        lo = (torch.pi/4 - (self.sifa + self.sbfa)/2) + (torch.pi/4 - beta/2)
        
        abc_denom = torch.tan(lo)
        abc_denom = torch.where(abc_denom.abs() < 1e-9, 1e-9 * torch.sign(abc_denom), abc_denom)
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
        z_denom1 = torch.where(z_denom1.abs() < 1e-9, 1e-9 * torch.sign(z_denom1), z_denom1)
        
        z_denom2 = (torch.sin(beta)*(torch.cos(beta)+torch.sin(beta)/z_denom1))
        z_denom2 = torch.where(z_denom2.abs() < 1e-9, 1e-9 * torch.sign(z_denom2), z_denom2)
        
        z = -L/3*((torch.cos(beta)+torch.sin(beta)/torch.tan(lo))*(-torch.square(torch.sin(beta)))) / z_denom2
        ADF = self.c_a * self.B * L
        
        W_denom = torch.tan(lo)
        W_denom = torch.where(W_denom.abs() < 1e-9, 1e-9 * torch.sign(W_denom), W_denom)
        W = self.uw * self.B * (0.5 * L * torch.sin(beta) * (L * torch.cos(beta) + L * torch.sin(beta) / W_denom))
        
        CF_denom = torch.sin(lo)
        CF_denom = torch.where(CF_denom.abs() < 1e-9, 1e-9 * torch.sign(CF_denom), CF_denom)
        CF = self.c * self.B * L * torch.sin(beta) / CF_denom
        
        ACF_denom = torch.tan(lo)
        ACF_denom = torch.where(ACF_denom.abs() < 1e-9, 1e-9 * torch.sign(ACF_denom), ACF_denom)
        ACF = 0.5 * c_s * (L**2) * torch.sin(beta) * (torch.cos(beta) + torch.sin(beta)/ACF_denom)
        
        SF_denom = torch.tan(lo)
        SF_denom = torch.where(SF_denom.abs() < 1e-9, 1e-9 * torch.sign(SF_denom), SF_denom)
        SF = 0.5 * K * self.uw * z * torch.tan(sifa_s) * torch.square(L) * torch.sin(beta) * (torch.cos(beta)+ torch.sin(beta) / SF_denom)

        R_s = (-ADF * torch.cos(beta+lo+self.sifa) + W * torch.sin(lo+self.sifa) + CF * torch.cos(self.sifa) + 2 * ACF * torch.cos(self.sifa) + 2 * SF * torch.cos(self.sifa)) / torch.sin(beta+lo+self.sbfa+self.sifa).clamp_min(1e-9)
        
        #center_z = torch.norm(PO, dim=-1) / 2
        center_z = L / 2
        p_n = 0.5 * self.uw * center_z*((1+K) + (1-K)*torch.cos(2*beta))

        p = self.cpf * p_n
        R_ps = (self.c_a + p_n * torch.tan(self.sbfa)) * self.B * L
        R_pt_denom = torch.tan(torch.tensor(self.alpha, device=self.device))
        R_pt_denom = torch.where(R_pt_denom.abs() < 1e-9, 1e-9 * torch.sign(R_pt_denom), R_pt_denom)
        R_pt = self.n * (p + (self.c_a + p * torch.tan(self.sifa)) * (1 / R_pt_denom)) * self.A_t
        
        R_p = R_ps + R_pt
        R_p = R_p * torch.cos(self.ang_tip_plate)
        R_s = torch.where(bkt[:,1] > self.soil_height, torch.zeros_like(R_s), R_s)
        R_p = torch.where(bkt[:,1] > self.soil_height, torch.zeros_like(R_p), R_p)
        
        # O,P,E: [...,2] (월드 or 로컬)
        d = PO / (PO.norm(dim=-1, keepdim=True).clamp_min(1e-9)) # plate dir
        n1 = torch.stack([-d[...,1], d[...,0]], dim=-1)         # rotate +90°
        n2 = -n1
        M  = 0.5*(bkt + self.P_cord)
        Q_in = (bkt + self.tip_pos2 + self.P_cord)/3.0         # interior ref
        use_n1 = ( (Q_in - M)*n1 ).sum(dim=-1) > 0.
        n = torch.where(use_n1.unsqueeze(-1), n1, n2)

        EO = self.tip_pos2 -bkt
        d1 = EO / (EO.norm(dim=-1, keepdim=True).clamp_min(1e-9))
        
        Fs = n * R_s.unsqueeze(-1)
        Fp = d1 * R_p.unsqueeze(-1)
        
        Fs_pos = (sep_plate_sur + bkt) / 2
        Fs_pos = torch.where(((torch.norm(sep_plate_sur - bkt, dim=-1) > L) |  torch.isnan(intersec_x)).unsqueeze(-1), (self.P_cord + bkt) / 2, Fs_pos)
        
        c, s = torch.cos(self.sbfa), torch.sin(self.sbfa)
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
        
        total_f = torch.cat((rotated_Fs, Fp), dim=1)
        total_pos = torch.cat((Fs_pos, Fp_pos), dim=1)
        
        Jw = self.get_jacobian()
        Fw = self.get_wrench(total_f, total_pos)
        
        tau_gen = torch.einsum('bij,bj->bi', Jw.transpose(1,2), Fw)  # (B,6+n)

        self.tau_base   = tau_gen[:, :6]   # (B,6)
        self.tau_joints = tau_gen[:, 6:]   # (B,n)
        
        return rotated_Fs, Fp

    # def A_of_theta(self, theta):
    #     # P on arc
    #     R = self.bucket_circle
    #     E = self.tip_pos2
    #     C = self.circle_mid
    #     O = self.tip_pos
    #     P = C + torch.stack([R*torch.cos(theta), R*torch.sin(theta)], dim=-1)
    #     EO = E - O
    #     PO = P - O
    #     Atr = 0.5*(EO[..., 0]*PO[..., 1] - EO[..., 1]*PO[..., 0]).abs()
    #     EC = E - C
    #     dtheta = theta - torch.atan2(EC[:,1], EC[:,0])
    #     dtheta = torch.clamp(dtheta, min=0.0)
    #     Aseg =  0.5 * (R**2) * (dtheta - torch.sin(dtheta))
    #     return Atr + Aseg, P

    # def solve_theta(self, iters=30, eps=1e-7):
    #     EC = self.tip_pos2 - self.circle_mid
    #     theta_E = torch.atan2(EC[:,1], EC[:,0])
    #     A_tgt = self.fill_ratio.clone()
    #     low  = theta_E
    #     high = theta_E + torch.pi

    #     while True:
    #         mid = 0.5*(low + high)
    #         A_mid, P = self.A_of_theta(mid)
    #         go_left = (A_mid >= A_tgt)  # monotone increasing
    #         high = torch.where(go_left, mid, high)
    #         low  = torch.where(go_left, low, mid)
    #         done = (high - low).abs() < eps
            
    #         self.max_L = torch.norm(P, dim=-1)
    #         self.P_cord = P
            
    #         if done.all(): break
        
    #     return 0.5*(low + high)  # θ*
    
    def get_intersect(self):
        self.solve_theta()
        eps = 1e-9

        bkt  = self.tip_pos   # (N,2): [x0, y0]
        bkt3 = self.tip_pos3  # (N,2): for gradient1

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
    
    def _cross2(self, a, b):
        return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]

    def _cbrt(self, x):
        return torch.sign(x) * torch.pow(torch.abs(x).clamp_min(1e-12), 1.0/3.0)
    
        # 기존의 A_of_theta는 그대로 사용
    def A_of_theta(self, theta):
        R = self.bucket_circle
        E = self.tip_pos2
        C = self.circle_mid
        O = self.tip_pos
        P = C + torch.stack([R*torch.cos(theta), R*torch.sin(theta)], dim=-1)
        EO = E - O
        PO = P - O
        Atr = 0.5*(self._cross2(EO, PO)).abs()
        EC = E - C
        dtheta = theta - torch.atan2(EC[:,1], EC[:,0])
        dtheta = torch.clamp(dtheta, min=0.0)
        Aseg = 0.5 * (R**2) * (dtheta - torch.sin(dtheta))
        return Atr + Aseg, P

    # === (1) θ_E 주변 3차 근사 계수(폐형) ===
    def _cubic_coeffs_initial(self):
        O, E, C = self.tip_pos, self.tip_pos2, self.circle_mid
        R = self.bucket_circle
        ex, ey = (E - O)[..., 0], (E - O)[..., 1]
        EC = E - C
        theta_E = torch.atan2(EC[:,1], EC[:,0])
        cE, sE = torch.cos(theta_E), torch.sin(theta_E)
        gamma = 0.5 * R * (ex*cE + ey*sE)                  # A'(θ_E)
        beta  = -0.25 * R * (ex*sE - ey*cE)                # 1/2 A''(θ_E)
        alpha = (R**2)/12.0 - gamma/6.0                    # 1/6 A'''(θ_E) 근사
        return alpha, beta, gamma, theta_E

    # === (2) θ_max 주변 3차 근사 계수(정확 도함수 기반) ===
    def _derivs_at(self, theta0):
        # A'(θ), A''(θ), A'''(θ)  (삼각형 항의 abs는 sign(cross)로 처리)
        R = self.bucket_circle
        O, E, C = self.tip_pos, self.tip_pos2, self.circle_mid

        cos0, sin0 = torch.cos(theta0), torch.sin(theta0)
        P0   = C + torch.stack([R*cos0, R*sin0], dim=-1)
        P1   = torch.stack([-R*sin0, R*cos0], dim=-1)      # dP/dθ
        P2   = torch.stack([-R*cos0, -R*sin0], dim=-1)     # d2P/dθ2
        P3   = -P1                                         # d3P/dθ3

        EO = E - O
        S0 = self._cross2(EO, P0 - O)                           # 부호 있는 cross
        sgn = torch.sign(S0)
        sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)

        # 삼각형 항 도함수들
        T1 = 0.5 * sgn * self._cross2(EO, P1)
        T2 = 0.5 * sgn * self._cross2(EO, P2)
        T3 = 0.5 * sgn * self._cross2(EO, P3)

        # 세그먼트 항 도함수들 (θ0 >= θ_E 가정)
        EC = E - C
        theta_E = torch.atan2(EC[:,1], EC[:,0])
        d0 = (theta0 - theta_E).clamp_min(0.0)
        S1 = 0.5 * (R**2) * (1.0 - torch.cos(d0))
        S2 = 0.5 * (R**2) * torch.sin(d0)
        S3 = 0.5 * (R**2) * torch.cos(d0)

        A1 = T1 + S1
        A2 = T2 + S2
        A3 = T3 + S3
        return A1, A2, A3

    # === (3) 카르다노: α x^3 + β x^2 + γ x - A = 0 의 실해 ===
    def _cardano(self, alpha, beta, gamma, A_rhs):
        a = alpha.clamp_min(1e-12)
        b = beta
        g = gamma
        # x = y - b/(3a);   y^3 + p y + q = 0
        shift = b / (3.0*a)
        p = (3*a*g - b*b) / (3*a*a)
        q = (27*a*a*(-A_rhs) - 9*a*b*g + 2*b**3) / (27*a**3)
        half_q = 0.5*q
        third_p = p/3.0
        Delta = half_q*half_q + third_p*third_p*third_p
        y = torch.empty_like(p)

        mask = (Delta >= 0)
        if mask.any():
            sqrtD = torch.sqrt(Delta[mask])
            y1 = self._cbrt(-half_q[mask] + sqrtD)
            y2 = self._cbrt(-half_q[mask] - sqrtD)
            y[mask] = y1 + y2
        if (~mask).any():
            pm = p[~mask]
            qm = q[~mask]
            r = torch.sqrt((-pm/3.0).clamp_min(1e-12))
            arg = (3*qm)/(2*pm) * torch.sqrt(-3.0/pm)
            arg = torch.clamp(arg, -1+1e-9, 1-1e-9)
            theta = torch.acos(arg) / 3.0
            y[~mask] = 2.0 * r * torch.cos(theta)

        x = y - shift
        return x

    # === (4) 반복 없이 θ 해를 반환 ===
    def solve_theta(self, use_tail_switch=0.6):
        """
        반복 없이 θ를 반환. 
        - 초기 구간( A_tgt <= use_tail_switch * A_max ): θ_E 주변 카르다노
        - 말기 구간: θ_max 주변(보완량 ΔA) 카르다노
        """
        # 기하 각들
        EC = self.tip_pos2 - self.circle_mid
        theta_E = torch.atan2(EC[:,1], EC[:,0])
        theta_max = theta_E + torch.pi  # 립 각도를 알고 있으면 그걸 쓰세요

        # 목표/경계 면적
        A_tgt = self.fill_ratio.clone()
        A_low, _ = self.A_of_theta(theta_E)
        A_max, _ = self.A_of_theta(theta_max)
        A_tgt = torch.minimum(torch.maximum(A_tgt, A_low), A_max)

        # 스위치: 초기/말기 구간 선택
        use_head = (A_tgt <= use_tail_switch * A_max)

        # --- 초기 구간: θ = θ_E + x (카르다노) ---
        alpha_h, beta_h, gamma_h, thetaE = self._cubic_coeffs_initial()
        x_h = self._cardano(alpha_h, beta_h, gamma_h, A_tgt)
        theta_head = thetaE + x_h

        # --- 말기 구간: y = θ_max - θ,  ΔA = A_max - A(θ) ≈ γ'y + β'y^2 + α'y^3 ---
        A1, A2, A3 = self._derivs_at(theta_max)   # A', A'', A'''
        alpha_t =  (1.0/6.0) * A3                 # α'  (y^3 계수)
        beta_t  = -(1.0/2.0) * A2                 # β'  (y^2 계수)
        gamma_t =  A1                              # γ'  (y   계수)
        dA = (A_max - A_tgt).clamp_min(0.0)
        y_t = self._cardano(alpha_t.clamp_min(1e-12), beta_t, gamma_t, dA)
        theta_tail = theta_max - y_t

        # --- 구간 합치기 ---
        theta = torch.where(use_head, theta_head, theta_tail)

        # 최종 P/기록
        P = self.circle_mid + torch.stack(
            [self.bucket_circle*torch.cos(theta),
             self.bucket_circle*torch.sin(theta)], dim=-1)
        self.P_cord = P
        self.max_L  = torch.norm(P, dim=-1)
        return theta


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
