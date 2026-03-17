from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2StandCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 48  # 不用高度图，去掉 187 个高度点
        episode_length_s = 10

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'       # 平地即可
        measure_heights = False   # 关掉高度图观测

    class commands(LeggedRobotCfg.commands):
        # 速度指令全部固定为 0
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.30]
        default_joint_angles = {
            'FL_hip_joint': 0.1,  'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1, 'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8, 'RL_thigh_joint': 1.0,
            'FR_thigh_joint': 0.8, 'RR_thigh_joint': 1.0,
            'FL_calf_joint': -1.5, 'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5, 'RR_calf_joint': -1.5,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 40.}
        damping = {'joint': 1.0}
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
        base_height_target = 0.30   # 站立目标高度
        soft_dof_pos_limit = 0.9
        only_positive_rewards = False  # 站立任务负奖励更有效

        class scales(LeggedRobotCfg.rewards.scales):
            # --- 关掉行走相关奖励 ---
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            feet_air_time = 0.0

            # --- 站立核心奖励 ---
            base_height = -5.0        # 惩罚高度偏离目标
            orientation = -2.0        # 惩罚机身倾斜
            ang_vel_xy = -0.5         # 惩罚翻滚/俯仰角速度
            lin_vel_z = -2.0          # 惩罚垂直方向抖动

            # --- 平滑/能耗惩罚 ---
            torques = -0.0002
            dof_vel = -0.01
            dof_acc = -2.5e-7
            action_rate = -0.01
            dof_pos_limits = -10.0

            # --- 站立时保持默认姿态 ---
            stand_still = -1.0        # 零指令时惩罚关节偏离默认值

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = False           # 初期训练先不推

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0

class Go2StandCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = 'stand_go2'
        max_iterations = 500
