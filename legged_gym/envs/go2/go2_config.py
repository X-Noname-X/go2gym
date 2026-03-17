from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2RoughCfg(LeggedRobotCfg):

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.30]          # Go2 站立高度约 0.30 m（A1 是 0.42）
        default_joint_angles = {
            'FL_hip_joint':  0.1,   'RL_hip_joint':  0.1,
            'FR_hip_joint': -0.1,   'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8,  'FR_thigh_joint': 0.8,
            'RL_thigh_joint': 1.0,  'RR_thigh_joint': 1.0,
            'FL_calf_joint': -1.5,  'FR_calf_joint': -1.5,
            'RL_calf_joint': -1.5,  'RR_calf_joint': -1.5,
        }

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class env(LeggedRobotCfg.env):
        num_observations = 48   # 去掉高度图后必须同步修改

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 40.}   # Go2 机身 6.9 kg，比 A1 重，需要更大刚度
        damping   = {'joint': 1.0}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"                          # URDF 中 foot link 的名字
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.30                   # 与 init pos[2] 一致
        class scales(LeggedRobotCfg.rewards.scales):
            # 行走核心奖励（继承默认值，按需调整）
            tracking_lin_vel =  1.0
            tracking_ang_vel =  0.5
            feet_air_time    =  1.0
            # 稳定性惩罚
            lin_vel_z   = -2.0
            ang_vel_xy  = -0.05
            orientation = -0.2                      # Go2 重心偏前，适当加大
            # 能耗/平滑惩罚
            torques     = -0.0002
            dof_vel     = -0.0
            dof_acc     = -2.5e-7
            action_rate = -0.01
            collision   = -1.0
            dof_pos_limits = -10.0

class Go2RoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'go2_walk'
        max_iterations = 1500
