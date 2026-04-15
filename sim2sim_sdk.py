"""
Sim2Sim（SDK 版，通用）：支持 walk（48维）和 gait/trot（52维）模型。

运行方式（两个终端）：
  终端1: cd /home/noname/unitree_mujoco/simulate_python && python unitree_mujoco.py
  终端2: conda activate unitree-rl && cd /home/noname/go2gym

  # 行走模型（48维，无步态 clock）：
  python sim2sim_sdk.py --mode walk --policy policy/go2/exported/policies/policy_1.pt

  # 对角步态模型（52维，含步态 clock）：
  python sim2sim_sdk.py --mode trot --policy policy/go2_trot/deploy
"""

import argparse
import time
import threading
import sys
import tty
import termios
import numpy as np
import torch

from unitree_sdk2py.core.channel import (
    ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, SportModeState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_ as LowCmd_default

# ─────────── DDS 配置 ───────────
DOMAIN_ID = 1
INTERFACE = "lo"
TOPIC_LOWSTATE  = "rt/lowstate"
TOPIC_HIGHSTATE = "rt/sportmodestate"
TOPIC_LOWCMD    = "rt/lowcmd"

# ─────────── 控制参数 ───────────
CTRL_DT      = 0.02
KP           = 40.0
KD           = 1.0
ACTION_SCALE = 0.25

# ─────────── 键盘控制命令（运行时可修改）───────────
# [lin_vel_x, lin_vel_y, ang_vel_yaw]
_cmd_lock   = threading.Lock()
_cmd        = np.array([0.0, 0.0, 0.0])   # 当前命令，按键按下时写入，松开归零
_cmd_speed  = 0.5                          # 当前预设速度幅值（m/s 或 rad/s）
_SPEED_STEP = 0.1                          # 每次调速的步长
_SPEED_MIN  = 0.1
_SPEED_MAX  = 2.0

# 按键定义（WASD 移动，QE 转向，+- 调速）
_KEYMAP = {
    'w': np.array([ 1.0,  0.0,  0.0]),   # 前进
    's': np.array([-1.0,  0.0,  0.0]),   # 后退
    'a': np.array([ 0.0,  1.0,  0.0]),   # 左平移
    'd': np.array([ 0.0, -1.0,  0.0]),   # 右平移
    'q': np.array([ 0.0,  0.0,  1.0]),   # 左转
    'e': np.array([ 0.0,  0.0, -1.0]),   # 右转
}

def _get_char_nonblock(fd, old_settings):
    """读取一个字符，超时返回 None（非阻塞轮询）。"""
    import select
    rlist, _, _ = select.select([fd], [], [], 0.05)
    if rlist:
        return sys.stdin.read(1)
    return None

def _keyboard_thread():
    """后台线程：按下按键 → 设置速度命令；松开（超时）→ 命令归零。"""
    global _cmd_speed
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    print("[键盘] W/S=前后  A/D=左右平移  Q/E=转向  ↑/↓(=/-)=调速  Ctrl+C退出")
    try:
        tty.setraw(fd)
        last_key_time = 0.0
        KEY_TIMEOUT = 0.15   # 超过此时间没有按键输入则命令归零

        while True:
            ch = _get_char_nonblock(fd, old_settings)
            now = time.perf_counter()

            if ch is not None:
                if ch in _KEYMAP:
                    with _cmd_lock:
                        _cmd[:] = _KEYMAP[ch] * _cmd_speed
                    last_key_time = now
                elif ch in ('=', '+'):   # 提速
                    _cmd_speed = min(_cmd_speed + _SPEED_STEP, _SPEED_MAX)
                    print(f"\r[键盘] 速度调整为 {_cmd_speed:.1f} m/s          ", end='', flush=True)
                elif ch in ('-', '_'):   # 减速
                    _cmd_speed = max(_cmd_speed - _SPEED_STEP, _SPEED_MIN)
                    print(f"\r[键盘] 速度调整为 {_cmd_speed:.1f} m/s          ", end='', flush=True)
                elif ch == '\x03':       # Ctrl+C
                    break
            else:
                # 超时未收到按键 → 停止
                if now - last_key_time > KEY_TIMEOUT:
                    with _cmd_lock:
                        _cmd[:] = 0.0
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def get_command() -> np.ndarray:
    """主循环调用：获取当前命令的线程安全副本。"""
    with _cmd_lock:
        return _cmd.copy()

# ─────────── 站立参数 ───────────
STANDUP_DURATION  = 2.0   # 插值到站立姿态所需秒数
STABILIZE_DURATION = 1.0  # 到位后等待稳定的秒数

# ─────────── 步态参数（trot，仅 mode=trot 时使用）───────────
GAIT_FREQ   = 3.0
GAIT_PHASE  = 0.5
GAIT_OFFSET = 0.0
GAIT_BOUND  = 0.0

# ─────────── 关节映射 ───────────
# SDK/MuJoCo sensor 顺序: FR(0-2), FL(3-5), RR(6-8), RL(9-11)
# legged_gym URDF 顺序:   FL(0-2), FR(3-5), RL(6-8), RR(9-11)
PERM = np.array([3, 4, 5,  0, 1, 2,  9, 10, 11,  6, 7, 8])

# 默认关节角（legged_gym 顺序）
DEFAULT_Q_LEGGED = np.array([
     0.1,  0.8, -1.5,   # FL
    -0.1,  0.8, -1.5,   # FR
     0.1,  1.0, -1.5,   # RL
    -0.1,  1.0, -1.5,   # RR
])
DEFAULT_Q_SDK = DEFAULT_Q_LEGGED[PERM]

# 观测缩放
OBS_SCALE_LIN_VEL = 2.0
OBS_SCALE_ANG_VEL = 0.25
OBS_SCALE_DOF_POS = 1.0
OBS_SCALE_DOF_VEL = 0.05
CMD_SCALES  = np.array([OBS_SCALE_LIN_VEL, OBS_SCALE_LIN_VEL, OBS_SCALE_ANG_VEL])
GRAVITY_VEC = np.array([0.0, 0.0, -1.0])


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """将世界坐标向量 v 旋转到体坐标系。"""
    w, x, y, z = q_wxyz
    q_vec = np.array([-x, -y, -z])
    t = 2.0 * np.cross(q_vec, v)
    return v + w * t + np.cross(q_vec, t)


class Controller:
    def __init__(self, policy_path: str, mode: str):
        self.mode = mode   # "walk" 或 "trot"

        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        print(f"[sim2sim] 策略已加载：{policy_path}  模式：{mode}")

        self._lock      = threading.Lock()
        self.low_state  = None
        self.high_state = None

        self.last_action  = np.zeros(12, dtype=np.float32)
        self.gait_index   = 0.0
        self.lin_vel_filt = np.zeros(3, dtype=np.float32)  # 低通滤波后的线速度

        # DDS 发布/订阅
        self.low_cmd_puber = ChannelPublisher(TOPIC_LOWCMD, LowCmd_)
        self.low_cmd_puber.Init()

        self.low_state_suber = ChannelSubscriber(TOPIC_LOWSTATE, LowState_)
        self.low_state_suber.Init(self._on_low_state, 10)

        self.high_state_suber = ChannelSubscriber(TOPIC_HIGHSTATE, SportModeState_)
        self.high_state_suber.Init(self._on_high_state, 10)

    def _on_low_state(self, msg):
        with self._lock:
            self.low_state = msg

    def _on_high_state(self, msg):
        with self._lock:
            self.high_state = msg

    def _compute_obs(self, low, high) -> np.ndarray:
        quat      = np.array(low.imu_state.quaternion)   # [w, x, y, z]
        gyro      = np.array(low.imu_state.gyroscope)    # 体坐标系角速度
        vel_world = np.array(high.velocity)              # 世界坐标系线速度
        lin_vel_raw = quat_rotate_inverse(quat, vel_world)
        # 低通滤波：α=0.1 平滑速度噪声（z方向跳动尤为明显）
        alpha = 0.1
        self.lin_vel_filt = (1 - alpha) * self.lin_vel_filt + alpha * lin_vel_raw
        lin_vel   = self.lin_vel_filt
        proj_grav = quat_rotate_inverse(quat, GRAVITY_VEC)

        # 关节状态重排为 legged_gym 顺序
        q_sdk  = np.array([low.motor_state[i].q  for i in range(12)])
        dq_sdk = np.array([low.motor_state[i].dq for i in range(12)])
        dof_pos = q_sdk[PERM]
        dof_vel = dq_sdk[PERM]

        # 公共部分（48维）
        obs = np.concatenate([
            lin_vel        * OBS_SCALE_LIN_VEL,                     # [0:3]
            gyro           * OBS_SCALE_ANG_VEL,                     # [3:6]
            proj_grav,                                                # [6:9]
            get_command()  * CMD_SCALES,                             # [9:12]
            (dof_pos - DEFAULT_Q_LEGGED) * OBS_SCALE_DOF_POS,       # [12:24]
            dof_vel        * OBS_SCALE_DOF_VEL,                     # [24:36]
            self.last_action,                                         # [36:48]
        ])

        # 步态模型额外追加 clock inputs（52维）
        if self.mode == "trot":
            g = self.gait_index
            raw = np.array([
                (g + GAIT_PHASE + GAIT_OFFSET + GAIT_BOUND) % 1.0,  # FL
                (g + GAIT_OFFSET) % 1.0,                             # FR
                (g + GAIT_BOUND)  % 1.0,                             # RL
                (g + GAIT_PHASE)  % 1.0,                             # RR
            ])
            clock = np.sin(2.0 * np.pi * raw)
            obs = np.concatenate([obs, clock])                       # [48:52]

        obs = np.clip(obs, -100.0, 100.0)   # 与训练 clip_observations 保持一致
        return obs.astype(np.float32)

    def _send_cmd(self, action_legged: np.ndarray):
        cmd = LowCmd_default()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        action_sdk = action_legged[PERM]
        target_q   = DEFAULT_Q_SDK + action_sdk * ACTION_SCALE
        for i in range(12):
            cmd.motor_cmd[i].q   = float(target_q[i])
            cmd.motor_cmd[i].dq  = 0.0
            cmd.motor_cmd[i].tau = 0.0
            cmd.motor_cmd[i].kp  = KP
            cmd.motor_cmd[i].kd  = KD
        self.low_cmd_puber.Write(cmd)

    def _send_standup_cmd(self, target_q_sdk: np.ndarray, kp: float = KP):
        """发送站立插值指令（不经过 ACTION_SCALE，直接指定目标角度）。"""
        cmd = LowCmd_default()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        for i in range(12):
            cmd.motor_cmd[i].q   = float(target_q_sdk[i])
            cmd.motor_cmd[i].dq  = 0.0
            cmd.motor_cmd[i].tau = 0.0
            cmd.motor_cmd[i].kp  = kp
            cmd.motor_cmd[i].kd  = KD
        self.low_cmd_puber.Write(cmd)

    def _standup(self):
        """平滑插值到默认站立姿态，然后等待稳定。"""
        with self._lock:
            low = self.low_state
        start_q = np.array([low.motor_state[i].q for i in range(12)])  # SDK 顺序

        steps = int(STANDUP_DURATION / CTRL_DT)
        print(f"[sim2sim] 站立阶段：{STANDUP_DURATION:.1f}s 插值到默认姿态...")
        for k in range(steps):
            alpha = (k + 1) / steps
            target = (1.0 - alpha) * start_q + alpha * DEFAULT_Q_SDK
            self._send_standup_cmd(target)
            time.sleep(CTRL_DT)

        # 稳定等待
        print(f"[sim2sim] 保持站立，等待 {STABILIZE_DURATION:.1f}s 稳定...")
        stab_steps = int(STABILIZE_DURATION / CTRL_DT)
        for _ in range(stab_steps):
            self._send_standup_cmd(DEFAULT_Q_SDK)
            time.sleep(CTRL_DT)

        print("[sim2sim] 站立稳定完成，切换到策略控制。")

    def run(self):
        print("[sim2sim] 等待仿真器就绪...")
        while True:
            with self._lock:
                if self.low_state is not None and self.high_state is not None:
                    break
            time.sleep(0.01)

        self._standup()

        # 启动键盘监听线程
        kb_thread = threading.Thread(target=_keyboard_thread, daemon=True)
        kb_thread.start()

        print("[sim2sim] 开始控制循环...")

        step = 0
        while True:
            t0 = time.perf_counter()

            with self._lock:
                low  = self.low_state
                high = self.high_state

            obs = self._compute_obs(low, high)
            with torch.no_grad():
                action = self.policy(
                    torch.from_numpy(obs).unsqueeze(0)
                ).squeeze(0).numpy()

            self._send_cmd(action)
            self.last_action = action.copy()

            if self.mode == "trot":
                self.gait_index = (self.gait_index + CTRL_DT * GAIT_FREQ) % 1.0

            step += 1
            if step % 47 == 0:
                h  = high.position[2]
                vx = np.dot(np.array(high.velocity),
                            quat_rotate_inverse(np.array(low.imu_state.quaternion),
                                                np.array([1, 0, 0])))
                g_str = f"  gait={self.gait_index:.2f}" if self.mode == "trot" else ""
                print(f"[{step:5d}] height={h:.3f}m  vel_x≈{vx:.2f}m/s{g_str}")

            elapsed = time.perf_counter() - t0
            if CTRL_DT - elapsed > 0:
                time.sleep(CTRL_DT - elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",   choices=["walk", "trot"], default="trot",
                        help="walk=48维无步态clock，trot=52维含步态clock")
    parser.add_argument("--policy", type=str,
                        default="policy/go2_trot/deploy",
                        help="TorchScript 策略文件路径")
    args = parser.parse_args()

    ChannelFactoryInitialize(DOMAIN_ID, INTERFACE)
    ctrl = Controller(args.policy, args.mode)
    ctrl.run()
