# =============================================================================
# cartpole_pid_direct.py
# Isaac Lab — CartPole PID Controller (Direct Workflow)
#
# Isaac Lab の DirectRLEnv を継承し、PID を「ActionTerm として埋め込む」のではなく
# 「エージェントが PID ゲインのスケールを出力し、_pre_physics_step で PID 演算する」
# 研究開発における最も一般的なパターンで実装する。
#
# 実行方法:
#   cd <IsaacLab-root>
#   ./isaaclab.sh -p /path/to/cartpole_pid_direct.py --num_envs 64
#   ./isaaclab.sh -p /path/to/cartpole_pid_direct.py --num_envs 1 --csv_out ./output/isaaclab_direct_pid.csv
#
# Isaac Lab における PID の位置づけ（Direct workflow）:
#   env.step(action) の action は「PID への外部指令」として解釈できる。
#   今回は action=0（固定ゲイン PID）で倒立振子を安定化する例として実装。
#   _pre_physics_step() 内で PID 計算 → _apply_action() で力を印加。
#
# CSV 出力:
#   Isaac Sim Standalone / MATLAB と列名統一
#   (time_s, cart_x_m, cart_v_ms, pole_rad, pole_deg, pole_omega_rads,
#    force_N, pid_up, pid_ui, pid_ud)
# =============================================================================

from __future__ import annotations
import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CartPole PID — Isaac Lab Direct")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_steps", type=int, default=7200,
                    help="1環境あたりの最大ステップ数 (7200=60s@120Hz)")
parser.add_argument("--csv_out", type=str, default="",
                    help="CSV 出力パス (空=出力なし) ※ num_envs=1 推奨")
parser.add_argument("--headless", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac Lab / Isaac Sim モジュール（AppLauncher の後）
import csv, os, time
import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG


# =============================================================================
# 1. 環境設定（DirectRLEnvCfg）
# =============================================================================
@configclass
class CartpolePIDEnvCfg(DirectRLEnvCfg):
    """
    PID 制御検証用 CartPole 環境設定。

    Isaac Lab Direct workflow の構成要素:
        decimation        : 物理ステップ / 制御ステップ比 (2 → 制御 60Hz, 物理 120Hz)
        episode_length_s  : 1エピソードの最大時間 [s]
        action_scale      : エージェント出力 [-1,1] → 実力 [N] の変換係数
        action_space      : エージェント出力次元数 (PID固定の場合は 0 or 1)
        observation_space : 観測ベクトル次元数 (x, ẋ, θ, θ̇) = 4
    """
    # ── MDP 設定
    decimation       = 2          # 制御は 60 Hz (物理 120 Hz の 1/2)
    episode_length_s = 60.0       # 最大 60 秒
    action_scale     = 1.0        # PID 固定制御: スケール不使用
    action_space     = 1          # ダミー (PID 内蔵のため実質不使用)
    observation_space = 4         # [cart_x, cart_v, pole_rad, pole_omega]
    state_space      = 0

    # ── シミュレーション設定
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # ── ロボット設定（公式 CartPole USD）
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"

    # ── シーン設定
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, env_spacing=4.0, replicate_physics=True
    )

    # ── エピソードリセット条件
    max_cart_pos            = 3.0     # カートが ±3m 以上で終了
    initial_pole_angle_range = [-0.25, 0.25]  # リセット時のポール初期角 [rad]


# =============================================================================
# 2. PID コントローラ（Torch ベクトル演算対応、並列環境対応）
# =============================================================================
class TorchCartPolePID:
    """
    GPU/CPU テンソル演算による並列 PID コントローラ。
    Isaac Lab の num_envs 並列環境を一括処理する。

    状態: (num_envs,) の torch.Tensor を受け取り、力を返す。

    制御則（全環境を同一ゲインで処理）:
        F = Kp_pole*θ + Ki_pole*∫θdt + Kd_pole*θ̇
          + Kp_cart*x  + Kd_cart*ẋ
    """
    def __init__(self,
                 num_envs: int,
                 device: str,
                 Kp_pole: float = 497.62,
                 Ki_pole: float = 5.0,
                 Kd_pole: float = 141.36,
                 Kp_cart: float = 31.62,
                 Kd_cart: float = 56.91,
                 integral_limit: float = 0.5,
                 force_limit: float = 500.0):
        self.Kp_pole = Kp_pole
        self.Ki_pole = Ki_pole
        self.Kd_pole = Kd_pole
        self.Kp_cart = Kp_cart
        self.Kd_cart = Kd_cart
        self.integral_limit = integral_limit
        self.force_limit = force_limit
        # 積分バッファ (num_envs,)
        self._integral = torch.zeros(num_envs, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """指定環境の積分バッファをリセット（エピソード終了時に呼ぶ）"""
        self._integral[env_ids] = 0.0

    def compute(self,
                cart_x: torch.Tensor,
                cart_v: torch.Tensor,
                theta:  torch.Tensor,
                omega:  torch.Tensor,
                dt:     float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            cart_x, cart_v, theta, omega : (num_envs,) テンソル
            dt : タイムステップ [s]
        Returns:
            force, up, ui, ud : (num_envs,) テンソル [N]
        """
        # 積分（クランプ法アンチワインドアップ）
        self._integral = torch.clamp(
            self._integral + theta * dt,
            -self.integral_limit, self.integral_limit
        )
        up = self.Kp_pole * theta
        ui = self.Ki_pole * self._integral
        ud = self.Kd_pole * omega
        u_cart = self.Kp_cart * cart_x + self.Kd_cart * cart_v
        force = torch.clamp(up + ui + ud + u_cart,
                            -self.force_limit, self.force_limit)
        return force, up, ui, ud


# =============================================================================
# 3. Direct RL 環境実装
# =============================================================================
class CartpolePIDEnv(DirectRLEnv):
    """
    CartPole PID 検証用 Direct RL 環境。

    Isaac Lab Direct workflow のフック関数:
        _setup_scene()          : ロボット・床・ライト生成
        _pre_physics_step()     : 物理ステップ前処理（PID計算）
        _apply_action()         : 関節に力を印加
        _get_observations()     : 観測ベクトルを返す
        _get_rewards()          : 報酬計算
        _get_dones()            : エピソード終了判定
        _reset_idx()            : 指定環境のリセット

    PID と Isaac Lab の関係:
        - action（エージェント出力）は今回「使用しない」（PID 固定制御）
        - _pre_physics_step で PID を計算し、self._pid_forces に保存
        - _apply_action で保存した力を joint effort として印加
        この分離設計は Isaac Lab の decimation メカニズムと一致する:
          decimation=2 の場合、_apply_action は物理ステップごと(120Hz)
          _pre_physics_step は制御ステップごと(60Hz) に呼ばれる
    """
    cfg: CartpolePIDEnvCfg

    def __init__(self, cfg: CartpolePIDEnvCfg,
                 render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # DOF インデックスの取得
        self._cart_dof_idx, _ = self.robot.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.robot.find_joints(self.cfg.pole_dof_name)

        # PID コントローラ（テンソル対応）
        self._pid = TorchCartPolePID(
            num_envs=self.num_envs,
            device=self.device,
        )

        # 力バッファ (num_envs, num_joints)
        self._pid_forces = torch.zeros(
            self.num_envs, self.robot.num_joints, device=self.device
        )

        # PID 内訳ログ（CSV出力用）
        self._last_up = torch.zeros(self.num_envs, device=self.device)
        self._last_ui = torch.zeros(self.num_envs, device=self.device)
        self._last_ud = torch.zeros(self.num_envs, device=self.device)

    # ── シーン構築 ────────────────────────────────────────────────
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.articulations["robot"] = self.robot
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0,
                                            color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ── 制御前処理（PID 計算）──────────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Isaac Lab の呼び出しタイミング: 制御ステップ（decimation = 2 → 60 Hz）

        actions は今回「RL エージェントの出力」であるが、PID 固定制御なので無視。
        将来的に:
          - actions をゲインの補正値として使う（アダプティブ PID）
          - actions を目標角度として使い、PID でトラッキングさせる
          ことで RL + PID の階層制御に拡張できる。
        """
        dt = self.cfg.sim.dt * self.cfg.decimation  # 制御周期 [s]

        # 状態取得（joint_pos/vel は (num_envs, num_joints) テンソル）
        cart_x = self.robot.data.joint_pos[:, self._cart_dof_idx[0]]
        cart_v = self.robot.data.joint_vel[:, self._cart_dof_idx[0]]
        theta  = self.robot.data.joint_pos[:, self._pole_dof_idx[0]]
        omega  = self.robot.data.joint_vel[:, self._pole_dof_idx[0]]

        # PID 計算（num_envs を一括処理）
        force, up, ui, ud = self._pid.compute(cart_x, cart_v, theta, omega, dt)

        # 力バッファに格納（cart joint のみ）
        self._pid_forces[:, self._cart_dof_idx[0]] = force
        self._pid_forces[:, self._pole_dof_idx[0]] = 0.0

        # ログ保存
        self._last_up = up
        self._last_ui = ui
        self._last_ud = ud

    # ── アクション印加 ────────────────────────────────────────────
    def _apply_action(self) -> None:
        """
        Isaac Lab の呼び出しタイミング: 物理ステップごと（120 Hz）

        _pre_physics_step で計算した力を joint effort として印加。
        decimation=2 なので、1制御ステップ間は同じ力が 2 物理ステップ適用される。
        → Isaac Sim Standalone の set_joint_efforts() 相当の操作
        """
        self.robot.set_joint_effort_target(self._pid_forces)

    # ── 観測ベクトル ──────────────────────────────────────────────
    def _get_observations(self) -> dict:
        """
        観測: [cart_x, cart_v, pole_rad, pole_omega]  shape: (num_envs, 4)

        Isaac Lab の RL エージェントはこの観測を受け取り action を出力する。
        PID 固定制御では観測は参照しないが、RL 連携時はここが Policy の入力になる。
        """
        obs = torch.cat([
            self.robot.data.joint_pos[:, self._cart_dof_idx],  # cart_x
            self.robot.data.joint_vel[:, self._cart_dof_idx],  # cart_v
            self.robot.data.joint_pos[:, self._pole_dof_idx],  # pole_rad
            self.robot.data.joint_vel[:, self._pole_dof_idx],  # pole_omega
        ], dim=-1)  # (num_envs, 4)
        return {"policy": obs}

    # ── 報酬 ─────────────────────────────────────────────────────
    def _get_rewards(self) -> torch.Tensor:
        """
        報酬: ポールが直立・カートが原点近くにいるほど高い。

        RL エージェントを使う場合はこの報酬シグナルで学習する。
        PID 固定制御では参照されないが、制御品質の指標として利用できる。
        """
        theta  = self.robot.data.joint_pos[:, self._pole_dof_idx[0]]
        cart_x = self.robot.data.joint_pos[:, self._cart_dof_idx[0]]
        reward = (1.0
                  - theta.pow(2)
                  - 0.1 * cart_x.pow(2))
        return reward

    # ── エピソード終了判定 ────────────────────────────────────────
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        終了条件:
          - ポール角度 > π/2（倒れ）
          - カート位置 > max_cart_pos（はみ出し）
          - タイムアウト（episode_length_s 超過）
        """
        theta  = self.robot.data.joint_pos[:, self._pole_dof_idx[0]]
        cart_x = self.robot.data.joint_pos[:, self._cart_dof_idx[0]]

        # 終了フラグ (num_envs,)
        terminated = (
            theta.abs() > math.pi / 2
        ) | (
            cart_x.abs() > self.cfg.max_cart_pos
        )
        # タイムアウト（truncated）
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    # ── リセット ─────────────────────────────────────────────────
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """
        指定環境をランダム初期状態にリセット。
        PID 積分バッファも同時にリセット。
        """
        if env_ids is None or len(env_ids) == 0:
            return
        super()._reset_idx(env_ids)

        # ランダム初期 joint 状態
        n = len(env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        # ポールをランダムに傾ける（±0.25 rad）
        joint_pos[:, self._pole_dof_idx[0]] = sample_uniform(
            self.cfg.initial_pole_angle_range[0],
            self.cfg.initial_pole_angle_range[1],
            (n,), device=self.device
        )
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # PID 積分バッファをリセット（重要: 前エピソードの積分残留を防ぐ）
        self._pid.reset(env_ids)


# =============================================================================
# 4. CSV ロガー（Standalone / MATLAB と列名統一）
# =============================================================================
class IsaacLabCsvLogger:
    """
    env_id=0 の環境データのみを記録する CSV ロガー。
    列名は Standalone / MATLAB と完全統一。
    """
    HEADER = [
        "time_s", "cart_x_m", "cart_v_ms",
        "pole_rad", "pole_deg", "pole_omega_rads",
        "force_N", "pid_up", "pid_ui", "pid_ud",
    ]

    def __init__(self, filepath: str):
        self._path = filepath
        self._buf  = []
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", newline="") as f:
            csv.writer(f).writerow(self.HEADER)

    def log(self, t, cart_x, cart_v, pole_rad, pole_omega,
            force, up, ui, ud):
        self._buf.append([
            f"{t:.6f}", f"{cart_x:.6f}", f"{cart_v:.6f}",
            f"{pole_rad:.6f}", f"{np.degrees(pole_rad):.4f}",
            f"{pole_omega:.6f}", f"{force:.4f}",
            f"{up:.4f}", f"{ui:.4f}", f"{ud:.4f}",
        ])
        if len(self._buf) >= 600:
            self._flush()

    def _flush(self):
        with open(self._path, "a", newline="") as f:
            csv.writer(f).writerows(self._buf)
        self._buf.clear()

    def close(self):
        self._flush()
        print(f"[CSV] 保存完了: {os.path.abspath(self._path)}")


# =============================================================================
# 5. メインループ
# =============================================================================
def main():
    # ── 環境インスタンス生成
    env_cfg = CartpolePIDEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = args_cli.max_steps / 120.0  # steps → sec

    env = CartpolePIDEnv(cfg=env_cfg, render_mode="rgb_array" if not args_cli.headless else None)

    # ── CSV ロガー（env_id=0 のデータのみ）
    logger = None
    if args_cli.csv_out:
        if args_cli.num_envs != 1:
            print("[WARN] CSV出力は --num_envs 1 推奨（env_id=0 のみ記録）")
        logger = IsaacLabCsvLogger(args_cli.csv_out)

    # ── ゼロアクション（PID 固定制御なのでエージェント出力は使わない）
    actions = torch.zeros(env.num_envs, env.cfg.action_space, device=env.device)

    print(f"\n[INFO] Isaac Lab Direct PID 開始")
    print(f"  num_envs   = {env.num_envs}")
    print(f"  physics_dt = {env.cfg.sim.dt:.6f} s  ({1/env.cfg.sim.dt:.0f} Hz)")
    print(f"  control_dt = {env.cfg.sim.dt * env.cfg.decimation:.6f} s  "
          f"({1/(env.cfg.sim.dt * env.cfg.decimation):.0f} Hz)")

    obs, _ = env.reset()
    step = 0
    t_start = time.time()

    while simulation_app.is_running() and step < args_cli.max_steps:

        # ── env.step() = _pre_physics_step → decimation × _apply_action → 観測更新
        obs, reward, terminated, truncated, info = env.step(actions)
        step += 1

        # ── ログ記録（env_id=0）
        if logger is not None:
            t_sim = step * env.cfg.sim.dt * env.cfg.decimation
            cart_x    = env.robot.data.joint_pos[0, env._cart_dof_idx[0]].item()
            cart_v    = env.robot.data.joint_vel[0, env._cart_dof_idx[0]].item()
            pole_rad  = env.robot.data.joint_pos[0, env._pole_dof_idx[0]].item()
            pole_omega = env.robot.data.joint_vel[0, env._pole_dof_idx[0]].item()
            force = env._pid_forces[0, env._cart_dof_idx[0]].item()
            up    = env._last_up[0].item()
            ui    = env._last_ui[0].item()
            ud    = env._last_ud[0].item()
            logger.log(t_sim, cart_x, cart_v, pole_rad, pole_omega,
                       force, up, ui, ud)

        # ── コンソール（制御ステップ換算、毎秒 = 60制御ステップ）
        if step % 60 == 0:
            t_sim = step * env.cfg.sim.dt * env.cfg.decimation
            # env_id=0 の代表値
            p0 = env.robot.data.joint_pos[0, env._pole_dof_idx[0]].item()
            x0 = env.robot.data.joint_pos[0, env._cart_dof_idx[0]].item()
            f0 = env._pid_forces[0, env._cart_dof_idx[0]].item()
            r0 = reward[0].item()
            n_done = terminated.sum().item()
            print(f"  ctrl={step:5d} t={t_sim:6.2f}s | "
                  f"θ={np.degrees(p0):+.3f}° x={x0:+.4f}m F={f0:+.1f}N | "
                  f"r={r0:.3f} | done={n_done}/{env.num_envs}")

    elapsed = time.time() - t_start
    if logger:
        logger.close()
    print(f"\n[INFO] 完了: {step} 制御ステップ / wall={elapsed:.1f}s")
    env.close()


if __name__ == "__main__":
    main()
