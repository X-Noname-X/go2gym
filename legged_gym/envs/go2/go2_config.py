from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2Cfg(LeggedRobotCfg):

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.34]
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
        num_observations = 52   # 48 基础观测 + 4 clock inputs（步态相位信号）

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 40.}
        damping   = {'joint': 1.0}
        action_scale = 0.25            # 减小步幅，避免迈步过宽
        decimation = 4
    
    class commands(LeggedRobotCfg.commands):
        curriculum = True
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-2.0, 2.0]
 
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"                          # URDF中foot link的名字
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1

    class gait(LeggedRobotCfg.gait):
        pass  # 默认 trot：phase=0.5, offset=0.0, bound=0.0

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.34
        kappa_gait_probs = 0.07
        gait_force_sigma = 50.0
        gait_vel_sigma   = 0.5
        only_positive_rewards = False  # 步态奖励本身为负值，不能裁剪为0
        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel =  2.5
            tracking_ang_vel =  0.5
            feet_air_time    =  2.0
            base_height      =  -30.0
            # 步态接触奖励
            tracking_contacts_shaped_force = 1.0
            tracking_contacts_shaped_vel   = 1.0
            # 稳定性惩罚
            lin_vel_z   = -2.0
            ang_vel_xy  = -0.5
            orientation = -5.0
            # 能耗/平滑惩罚
            torques     = -0.0002
            dof_vel     = -0.0
            dof_pos     = -0.0
            hip_pos     = -10.0    # 专项惩罚髋关节外展，不影响大腿/小腿运动
            dof_acc     = -2.5e-7
            action_rate = -0.01
            collision   = -1.0
            dof_pos_limits = -10.0

class Go2CfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'go2'
        max_iterations = 3000

# ────────────── 单步态训练 Cfg ──────────────
# 步态参数对照：
#   trot  (对角步): phase=0.5, offset=0.0, bound=0.0
#   pace  (侧  步): phase=0.0, offset=0.5, bound=0.0
#   bound (跳  步): phase=0.0, offset=0.0, bound=0.5
#   pronk (全腾跳): phase=0.5, offset=0.5, bound=0.5

class Go2TrotCfg(Go2Cfg):
    class gait(Go2Cfg.gait):
        phase  = 0.5
        offset = 0.0
        bound  = 0.0

class Go2PaceCfg(Go2Cfg):
    class gait(Go2Cfg.gait):
        phase  = 0.0
        offset = 0.5
        bound  = 0.0

class Go2BoundCfg(Go2Cfg):
    class gait(Go2Cfg.gait):
        phase  = 0.0
        offset = 0.0
        bound  = 0.5

class Go2PronkCfg(Go2Cfg):
    class gait(Go2Cfg.gait):
        phase  = 0.5
        offset = 0.5
        bound  = 0.5

class Go2GaitCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'go2_trot'
        max_iterations = 1000
