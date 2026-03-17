from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2RunCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 48
        episode_length_s = 10

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.30]  # 机身质心初始高度 [m]
        default_joint_angles = {
            'FL_hip_joint':  0.1,   'RL_hip_joint':  0.1,
            'FR_hip_joint': -0.1,   'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8,  'FR_thigh_joint': 0.8,
            'RL_thigh_joint': 1.0,  'RR_thigh_joint': 1.0,
            'FL_calf_joint': -1.5,  'FR_calf_joint': -1.5,
            'RL_calf_joint': -1.5,  'RR_calf_joint': -1.5,
        }

    class commands(LeggedRobotCfg.commands):
        heading_command = False  # 直接指定 yaw 角速度，不用 heading 模式
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x   = [0.5, 1.0]   # 向前跑 1.5~2.0 m/s
            lin_vel_y   = [0.0, 0.0]   # 不侧移
            ang_vel_yaw = [0.0, 0.0]   # 不转向，直线跑
            heading     = [0.0, 0.0]

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 40.}   # Go2 机身 6.9 kg，需要更大刚度
        damping   = {'joint': 1.0}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.30
        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 2.0   # 提高速度跟踪权重（跑步任务）
            tracking_ang_vel = 0.5
            feet_air_time    = 1.0
            lin_vel_z        = -2.0
            ang_vel_xy       = -0.05
            orientation      = -0.2  # Go2 重心有前向偏移，适当约束
            torques          = -0.0002
            dof_vel          = -0.0
            dof_acc          = -2.5e-7
            action_rate      = -0.01
            collision        = -1.0
            dof_pos_limits   = -10.0

class Go2RunCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'go2_run'
        max_iterations = 1500
