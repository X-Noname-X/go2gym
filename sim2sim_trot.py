"""
Sim2Sim: 在 MuJoCo 中运行 go2gym 训练的 trot 步态策略

依赖：
  conda activate unitree-rl   (含 mujoco, torch)

运行：
  cd /home/noname/go2gym
  python sim2sim_trot.py
"""

import time
import mujoco
import mujoco.viewer
import numpy as np
import torch

# ─────────── 路径 ───────────
POLICY_PATH = "policy/go2_gait/exported/policies/policy_1.pt"
SCENE_PATH  = "/home/noname/unitree_mujoco/unitree_robots/go2/scene.xml"

# ─────────── 仿真参数 ───────────
SIM_DT     = 0.005   # 200 Hz
DECIMATION = 4       # 策略 50 Hz（与训练 decimation=4 一致）
CTRL_DT    = SIM_DT * DECIMATION

# ─────────── 机器人参数 ───────────
KP = 40.0
KD = 1.0
ACTION_SCALE = 0.25

# 速度指令 [lin_vel_x, lin_vel_y, ang_vel_yaw]
COMMAND = np.array([0.5, 0.0, 0.0])

# 步态参数（trot）
GAIT_FREQ     = 3.0
GAIT_PHASE    = 0.5   # FL 相位
GAIT_OFFSET   = 0.0   # FR 相位
GAIT_BOUND    = 0.0   # RL 相位
# RR 相位 = GAIT_PHASE
# 各腿原始相位：FL=g+phase+offset+bound, FR=g+offset, RL=g+bound, RR=g+phase

# ─────────── 关节映射 ───────────
# Isaac Gym URDF 顺序: FL(0-2), FR(3-5), RL(6-8), RR(9-11)
# MuJoCo 顺序:        FR(0-2), FL(3-5), RR(6-8), RL(9-11)
# 互换置换（对称，读写共用）
PERM = np.array([3, 4, 5,  0, 1, 2,  9, 10, 11,  6, 7, 8])

# 默认关节角（legged_gym 顺序: FL, FR, RL, RR）
DEFAULT_Q_LEGGED = np.array([
     0.1,  0.8, -1.5,   # FL: hip, thigh, calf
    -0.1,  0.8, -1.5,   # FR
     0.1,  1.0, -1.5,   # RL
    -0.1,  1.0, -1.5,   # RR
])
DEFAULT_Q_MJ = DEFAULT_Q_LEGGED[PERM]   # MuJoCo 顺序

# 观测缩放
OBS_SCALE_LIN_VEL = 2.0
OBS_SCALE_ANG_VEL = 0.25
OBS_SCALE_DOF_POS = 1.0
OBS_SCALE_DOF_VEL = 0.05
CMD_SCALES = np.array([OBS_SCALE_LIN_VEL, OBS_SCALE_LIN_VEL, OBS_SCALE_ANG_VEL])

GRAVITY_VEC = np.array([0.0, 0.0, -1.0])


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """将世界坐标向量 v 旋转到体坐标系（使用四元数共轭）。
    q_wxyz: [w, x, y, z]"""
    w, x, y, z = q_wxyz
    q_vec = np.array([-x, -y, -z])   # 共轭虚部
    t = 2.0 * np.cross(q_vec, v)
    return v + w * t + np.cross(q_vec, t)


def compute_obs(sd: np.ndarray, last_action: np.ndarray,
                gait_index: float) -> np.ndarray:
    """构建 52 维观测向量，与 legged_gym compute_observations() 完全对应。

    sensordata 布局：
      [0:12]   关节位置（MuJoCo 顺序: FR/FL/RR/RL）
      [12:24]  关节速度（同上）
      [24:36]  关节力矩
      [36:40]  imu_quat  [w, x, y, z]
      [40:43]  imu_gyro  体坐标系角速度
      [43:46]  imu_acc
      [46:49]  frame_pos
      [49:52]  frame_vel  世界坐标系线速度
    """
    quat     = sd[36:40]           # [w, x, y, z]
    gyro     = sd[40:43]           # 体坐标系角速度
    vel_world = sd[49:52]          # 世界坐标系线速度

    # 线速度旋转到体坐标系
    lin_vel_body = quat_rotate_inverse(quat, vel_world)

    # 重力投影到体坐标系
    proj_gravity = quat_rotate_inverse(quat, GRAVITY_VEC)

    # 关节位置/速度，重排为 legged_gym 顺序
    dof_pos = sd[PERM]             # [12]
    dof_vel = sd[12 + PERM]        # [12]

    # 步态 clock inputs（sin 信号，4 条腿）
    raw = np.array([
        (gait_index + GAIT_PHASE + GAIT_OFFSET + GAIT_BOUND) % 1.0,  # FL
        (gait_index + GAIT_OFFSET) % 1.0,                             # FR
        (gait_index + GAIT_BOUND)  % 1.0,                             # RL
        (gait_index + GAIT_PHASE)  % 1.0,                             # RR
    ])
    clock = np.sin(2.0 * np.pi * raw)

    obs = np.concatenate([
        lin_vel_body  * OBS_SCALE_LIN_VEL,          # 3
        gyro          * OBS_SCALE_ANG_VEL,           # 3
        proj_gravity,                                 # 3
        COMMAND       * CMD_SCALES,                   # 3
        (dof_pos - DEFAULT_Q_LEGGED) * OBS_SCALE_DOF_POS,  # 12
        dof_vel       * OBS_SCALE_DOF_VEL,           # 12
        last_action,                                  # 12
        clock,                                        # 4
    ])                                                # 总计 52
    return obs.astype(np.float32)


def apply_action(mj_data: mujoco.MjData, action_legged: np.ndarray) -> None:
    """将 legged_gym 顺序的 action 转换为 MuJoCo 顺序，计算 PD 力矩并写入 ctrl。"""
    sd = mj_data.sensordata
    current_q  = sd[0:12]
    current_dq = sd[12:24]

    # action → MuJoCo 顺序 → 关节目标位置
    action_mj = action_legged[PERM]
    target_q  = DEFAULT_Q_MJ + action_mj * ACTION_SCALE

    # PD 控制
    torque = KP * (target_q - current_q) + KD * (0.0 - current_dq)
    mj_data.ctrl[:] = torque


def main():
    # 加载策略
    policy = torch.jit.load(POLICY_PATH)
    policy.eval()
    print(f"[sim2sim] 策略已加载：{POLICY_PATH}")

    # 加载 MuJoCo 模型
    mj_model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    mj_model.opt.timestep = SIM_DT
    mj_data  = mujoco.MjData(mj_model)

    # 初始化关节到站立姿态
    # qpos: [free joint pos(3) + quat(4), joints(12)]
    mj_data.qpos[3]  = 1.0                          # 四元数 w=1
    mj_data.qpos[7:] = DEFAULT_Q_MJ                 # 关节角
    mj_data.qpos[2]  = 0.34                          # 初始高度
    mujoco.mj_forward(mj_model, mj_data)

    last_action = np.zeros(12, dtype=np.float32)
    gait_index  = 0.0
    step_count  = 0

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -15
        print("[sim2sim] MuJoCo viewer 已启动，开始运行...")

        while viewer.is_running():
            step_start = time.perf_counter()

            # 仿真 DECIMATION 步
            for _ in range(DECIMATION):
                mujoco.mj_step(mj_model, mj_data)

            # 构建观测
            obs = compute_obs(mj_data.sensordata.copy(), last_action, gait_index)
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)  # (1, 52)

            # 策略推理
            with torch.no_grad():
                action_tensor = policy(obs_tensor)
            action = action_tensor.squeeze(0).numpy()

            # 应用动作
            apply_action(mj_data, action)
            last_action = action.copy()

            # 更新步态相位
            gait_index = (gait_index + CTRL_DT * GAIT_FREQ) % 1.0

            # 同步 viewer
            viewer.sync()

            step_count += 1
            if step_count % 47 == 0:
                sd = mj_data.sensordata
                height = mj_data.qpos[2]
                vel_x  = sd[49]   # 世界坐标系 x 方向速度（近似）
                print(f"[{step_count:5d}] height={height:.3f}m  vel_x≈{vel_x:.2f}m/s"
                      f"  gait={gait_index:.2f}")

            # 实时控制
            elapsed = time.perf_counter() - step_start
            sleep_t = CTRL_DT - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)


if __name__ == "__main__":
    main()
