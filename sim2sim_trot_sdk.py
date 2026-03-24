"""
Sim2Sim（SDK 版）：通过 unitree_sdk2py 与 unitree_mujoco 通信，
测试完整部署链路（与真机接口完全一致）。

运行方式（两个终端）：
  终端1: cd /home/noname/unitree_mujoco/simulate_python && python unitree_mujoco.py
  终端2: conda activate unitree-rl
         cd /home/noname/go2gym && python sim2sim_trot_sdk.py

DDS 配置与 unitree_mujoco/simulate_python/config.py 一致：
  DOMAIN_ID = 1, INTERFACE = "lo"
"""

import time
import threading
import numpy as np
import torch

from unitree_sdk2py.core.channel import (
    ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, SportModeState_
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_ as LowCmd_default,
)

# ─────────── 路径 ───────────
POLICY_PATH = "policy/go2_gait/exported/policies/policy_1.pt"

# ─────────── DDS 配置（与 unitree_mujoco config.py 一致）───────────
DOMAIN_ID = 1
INTERFACE = "lo"

TOPIC_LOWSTATE  = "rt/lowstate"
TOPIC_HIGHSTATE = "rt/sportmodestate"
TOPIC_LOWCMD    = "rt/lowcmd"

# ─────────── 控制参数 ───────────
CTRL_DT      = 0.02     # 50 Hz（decimation=4 × sim_dt=0.005）
KP           = 40.0
KD           = 1.0
ACTION_SCALE = 0.25

# 速度指令 [lin_vel_x, lin_vel_y, ang_vel_yaw]
COMMAND = np.array([0.5, 0.0, 0.0])

# 步态参数（trot）
GAIT_FREQ   = 3.0
GAIT_PHASE  = 0.5   # FL 相位偏移
GAIT_OFFSET = 0.0   # FR 相位偏移
GAIT_BOUND  = 0.0   # RL 相位偏移
# RR 相位 = GAIT_PHASE

# ─────────── 关节映射 ───────────
# SDK/MuJoCo 顺序: FR(0-2), FL(3-5), RR(6-8), RL(9-11)
# legged_gym 顺序: FL(0-2), FR(3-5), RL(6-8), RR(9-11)
# 互换置换（对称）
PERM = np.array([3, 4, 5,  0, 1, 2,  9, 10, 11,  6, 7, 8])

# 默认关节角（legged_gym 顺序: FL, FR, RL, RR）
DEFAULT_Q_LEGGED = np.array([
     0.1,  0.8, -1.5,   # FL
    -0.1,  0.8, -1.5,   # FR
     0.1,  1.0, -1.5,   # RL
    -0.1,  1.0, -1.5,   # RR
])
DEFAULT_Q_SDK = DEFAULT_Q_LEGGED[PERM]   # SDK/MuJoCo 顺序

# 观测缩放
OBS_SCALE_LIN_VEL = 2.0
OBS_SCALE_ANG_VEL = 0.25
OBS_SCALE_DOF_POS = 1.0
OBS_SCALE_DOF_VEL = 0.05
CMD_SCALES    = np.array([OBS_SCALE_LIN_VEL, OBS_SCALE_LIN_VEL, OBS_SCALE_ANG_VEL])
GRAVITY_VEC   = np.array([0.0, 0.0, -1.0])


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """将世界坐标向量 v 旋转到体坐标系（使用四元数共轭）。"""
    w, x, y, z = q_wxyz
    q_vec = np.array([-x, -y, -z])
    t = 2.0 * np.cross(q_vec, v)
    return v + w * t + np.cross(q_vec, t)


class TrotController:
    def __init__(self):
        self.policy = torch.jit.load(POLICY_PATH)
        self.policy.eval()
        print(f"[sdk] 策略已加载：{POLICY_PATH}")

        # 最新状态（线程共享，用锁保护）
        self._lock         = threading.Lock()
        self.low_state     = None
        self.high_state    = None

        # 控制器状态
        self.last_action   = np.zeros(12, dtype=np.float32)
        self.gait_index    = 0.0

        # LowCmd 发布器
        self.low_cmd_puber = ChannelPublisher(TOPIC_LOWCMD, LowCmd_)
        self.low_cmd_puber.Init()

        # LowState 订阅器
        self.low_state_suber = ChannelSubscriber(TOPIC_LOWSTATE, LowState_)
        self.low_state_suber.Init(self._on_low_state, 10)

        # SportModeState（包含机体速度）订阅器
        self.high_state_suber = ChannelSubscriber(TOPIC_HIGHSTATE, SportModeState_)
        self.high_state_suber.Init(self._on_high_state, 10)

    def _on_low_state(self, msg: LowState_):
        with self._lock:
            self.low_state = msg

    def _on_high_state(self, msg: SportModeState_):
        with self._lock:
            self.high_state = msg

    def _compute_obs(self, low: LowState_, high: SportModeState_) -> np.ndarray:
        """构建 52 维观测，与 legged_gym compute_observations() 完全对应。"""
        # IMU 数据（来自 LowState）
        quat = np.array(low.imu_state.quaternion)   # [w, x, y, z]
        gyro = np.array(low.imu_state.gyroscope)     # 体坐标系角速度

        # 机体线速度：high_state.velocity 是世界坐标系，旋转到体坐标系
        vel_world = np.array(high.velocity)           # [vx, vy, vz] 世界系
        lin_vel   = quat_rotate_inverse(quat, vel_world)

        # 重力在体坐标系的投影
        proj_gravity = quat_rotate_inverse(quat, GRAVITY_VEC)

        # 关节位置/速度，从 SDK 顺序重排到 legged_gym 顺序
        q_sdk  = np.array([low.motor_state[i].q  for i in range(12)])
        dq_sdk = np.array([low.motor_state[i].dq for i in range(12)])
        dof_pos = q_sdk[PERM]
        dof_vel = dq_sdk[PERM]

        # 步态 clock inputs（sin 信号，4 条腿，顺序 FL/FR/RL/RR）
        g = self.gait_index
        raw = np.array([
            (g + GAIT_PHASE + GAIT_OFFSET + GAIT_BOUND) % 1.0,  # FL
            (g + GAIT_OFFSET) % 1.0,                             # FR
            (g + GAIT_BOUND)  % 1.0,                             # RL
            (g + GAIT_PHASE)  % 1.0,                             # RR
        ])
        clock = np.sin(2.0 * np.pi * raw)

        obs = np.concatenate([
            lin_vel       * OBS_SCALE_LIN_VEL,               # [0:3]
            gyro          * OBS_SCALE_ANG_VEL,               # [3:6]
            proj_gravity,                                      # [6:9]
            COMMAND       * CMD_SCALES,                        # [9:12]
            (dof_pos - DEFAULT_Q_LEGGED) * OBS_SCALE_DOF_POS, # [12:24]
            dof_vel       * OBS_SCALE_DOF_VEL,               # [24:36]
            self.last_action,                                  # [36:48]
            clock,                                             # [48:52]
        ])
        return obs.astype(np.float32)

    def _send_cmd(self, action_legged: np.ndarray):
        """将 legged_gym 顺序的动作转换并发送 LowCmd（PD 目标位置）。"""
        cmd = LowCmd_default()
        # 固定模式字节（参考 unitree 示例）
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF

        # 转换到 SDK 顺序
        action_sdk = action_legged[PERM]
        target_q   = DEFAULT_Q_SDK + action_sdk * ACTION_SCALE

        for i in range(12):
            cmd.motor_cmd[i].q   = float(target_q[i])
            cmd.motor_cmd[i].dq  = 0.0
            cmd.motor_cmd[i].tau = 0.0
            cmd.motor_cmd[i].kp  = KP
            cmd.motor_cmd[i].kd  = KD

        self.low_cmd_puber.Write(cmd)

    def run(self):
        print("[sdk] 等待仿真器就绪...")
        # 等待第一条消息
        while True:
            with self._lock:
                if self.low_state is not None and self.high_state is not None:
                    break
            time.sleep(0.01)
        print("[sdk] 收到状态，开始控制循环...")

        step = 0
        while True:
            t0 = time.perf_counter()

            # 安全复制当前状态
            with self._lock:
                low  = self.low_state
                high = self.high_state

            # 构建观测 → 推理 → 发送指令
            obs = self._compute_obs(low, high)
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            with torch.no_grad():
                action = self.policy(obs_t).squeeze(0).numpy()

            self._send_cmd(action)
            self.last_action = action.copy()

            # 更新步态相位
            self.gait_index = (self.gait_index + CTRL_DT * GAIT_FREQ) % 1.0

            step += 1
            if step % 47 == 0:
                h  = high.position[2]
                vx = np.array(high.velocity) @ quat_rotate_inverse(
                    np.array(low.imu_state.quaternion), np.array([1,0,0]))
                print(f"[{step:5d}] height={h:.3f}m  vel_x≈{vx:.2f}m/s"
                      f"  gait={self.gait_index:.2f}")

            # 实时控制（50 Hz）
            elapsed = time.perf_counter() - t0
            sleep_t = CTRL_DT - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)


if __name__ == "__main__":
    ChannelFactoryInitialize(DOMAIN_ID, INTERFACE)
    ctrl = TrotController()
    ctrl.run()
