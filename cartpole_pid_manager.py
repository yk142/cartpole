# =============================================================================
# cartpole_pid_manager.py
# Isaac Lab — CartPole PID Controller (Manager-Based Workflow)
#
# Manager-based workflow での PID 実装パターン:
#   ① カスタム ActionTerm として PID を実装する方法
#      → RL エージェントの action を「PID への目標値」として受け取る
#   ② 標準の JointEffortActionCfg を使い、外部ループで PID を呼ぶ方法
#
# ここでは ① を実装する。PID ActionTerm は:
#   - action  = 目標ポール角度 (rad)  → PID の setpoint として使用
#   - または action = ゼロ → 標準の直立安定化 PID（固定ゲイン）
#
# 実行方法:
#   cd <IsaacLab-root>
#   ./isaaclab.sh -p /path/to/cartpole_pid_manager.py --num_envs 64
#
# =============================================================================

from __future__ import annotations
import argparse, math, time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CartPole PID — Isaac Lab Manager")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_steps", type=int, default=3600,
                    help="制御ステップ数 (3600=60s@60Hz)")
parser.add_argument("--headless", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time
import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import (
    ActionTerm, ActionTermCfg,
    ObservationGroupCfg as ObsGroup,
    ObservationManager,
    ObservationTermCfg as ObsTerm,
    EventTermCfg as EventTerm,
    EventManager,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG
import isaaclab.envs.mdp as mdp


# =============================================================================
# 1. カスタム PID ActionTerm
# =============================================================================
class CartpolePIDActionTerm(ActionTerm):
    """
    Isaac Lab Manager-Based 環境用 カスタム PID ActionTerm。

    Isaac Lab の ActionTerm インタフェース:
        __init__(cfg, env)   : 初期化
        action_dim           : エージェント出力次元数 (property)
        raw_actions          : 現在のエージェント出力 (property)
        processed_actions    : 処理済みアクション (property)
        process_actions(act) : エージェント出力を受け取り前処理
        apply_actions()      : 関節に命令を送る（物理ステップごとに呼ばれる）
        reset(env_ids)       : エピソードリセット時に呼ばれる

    エージェント出力 action (dim=1):
        - 今回は「目標ポール角度 [rad]」として解釈
        - action=0 → 直立安定化 (通常の PID)
        - action=θ_ref → θ_ref に追従する PID

    PID 計算位置:
        apply_actions() 内で毎物理ステップ実行。
        decimation が 2 の場合:
          process_actions() は 60Hz (制御ステップ)
          apply_actions()   は 120Hz (物理ステップ)
        → 制御力は 120Hz で更新される（force hold は自動）
    """
    cfg: "CartpolePIDActionTermCfg"

    def __init__(self, cfg: "CartpolePIDActionTermCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # DOF インデックス
        self._robot: Articulation = env.scene.articulations[cfg.asset_name]
        cart_ids, _ = self._robot.find_joints("slider_to_cart")
        pole_ids, _ = self._robot.find_joints("cart_to_pole")
        self._cart_idx = cart_ids[0]
        self._pole_idx = pole_ids[0]

        # PID パラメータ
        self.Kp_pole = cfg.Kp_pole
        self.Ki_pole = cfg.Ki_pole
        self.Kd_pole = cfg.Kd_pole
        self.Kp_cart = cfg.Kp_cart
        self.Kd_cart = cfg.Kd_cart
        self.integral_limit = cfg.integral_limit
        self.force_limit = cfg.force_limit

        # テンソルバッファ
        self._integral = torch.zeros(env.num_envs, device=env.device)
        self._forces = torch.zeros(
            env.num_envs, self._robot.num_joints, device=env.device
        )
        # エージェント目標値（setpoint）
        self._theta_ref = torch.zeros(env.num_envs, device=env.device)
        self._raw_actions   = torch.zeros(env.num_envs, self.action_dim, device=env.device)
        self._proc_actions  = torch.zeros(env.num_envs, self.action_dim, device=env.device)

    # ── ActionTerm インタフェース ─────────────────────────────────
    @property
    def action_dim(self) -> int:
        return 1   # 目標ポール角度 1次元

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._proc_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """
        エージェント出力を受け取る（制御ステップ = 60Hz）。
        actions: (num_envs, 1)  目標ポール角度 [rad]
        """
        self._raw_actions[:] = actions
        # スケーリング（cfg.scale に準拠、デフォルト 0.1 rad = 約 5.7°）
        self._theta_ref[:] = (actions[:, 0] * self.cfg.scale).clamp(-0.5, 0.5)
        self._proc_actions[:] = actions

    def apply_actions(self) -> None:
        """
        PID 演算 + joint effort 印加（物理ステップ = 120Hz）。

        Isaac Lab の decimation=2 環境では apply_actions() が 2回呼ばれる間に
        process_actions() が 1回呼ばれる。theta_ref は前回の値を保持する。
        """
        dt = self._env.physics_dt   # 物理ステップ時間 [s]

        # 状態取得
        cart_x = self._robot.data.joint_pos[:, self._cart_idx]
        cart_v = self._robot.data.joint_vel[:, self._cart_idx]
        theta  = self._robot.data.joint_pos[:, self._pole_idx]
        omega  = self._robot.data.joint_vel[:, self._pole_idx]

        # 追従誤差（θ_ref - θ）
        # 固定ゲイン直立安定化: theta_ref=0 → error = -theta
        error = self._theta_ref - theta

        # PID 計算
        self._integral = torch.clamp(
            self._integral + error * dt,
            -self.integral_limit, self.integral_limit
        )
        up = self.Kp_pole * error
        ui = self.Ki_pole * self._integral
        ud = self.Kd_pole * (-omega)   # 微分はポール角速度の逆符号
        u_cart = self.Kp_cart * cart_x + self.Kd_cart * cart_v
        force = torch.clamp(up + ui + ud + u_cart,
                            -self.force_limit, self.force_limit)

        self._forces[:, self._cart_idx] = force
        self._forces[:, self._pole_idx] = 0.0
        self._robot.set_joint_effort_target(self._forces)

    def reset(self, env_ids: torch.Tensor) -> None:
        """エピソードリセット時に積分・参照値をクリア"""
        self._integral[env_ids] = 0.0
        self._theta_ref[env_ids] = 0.0


@configclass
class CartpolePIDActionTermCfg(ActionTermCfg):
    """CartpolePIDActionTerm の設定クラス"""
    class_type: type[ActionTerm] = CartpolePIDActionTerm
    asset_name:  str   = "robot"

    # PID ゲイン（LQR由来、MATLAB/Standalone と統一）
    Kp_pole: float = 497.62
    Ki_pole: float = 5.0
    Kd_pole: float = 141.36
    Kp_cart: float = 31.62
    Kd_cart: float = 56.91
    integral_limit: float = 0.5
    force_limit:    float = 500.0

    # エージェント出力のスケール（action=1 → 目標角 scale rad）
    scale: float = 0.0  # 0.0 = 直立固定制御（action 無効）


# =============================================================================
# 2. シーン設定
# =============================================================================
@configclass
class CartpolePIDSceneCfg(InteractiveSceneCfg):
    ground = sim_utils.GroundPlaneCfg(size=(100.0, 100.0))
    robot: ArticulationCfg = CARTPOLE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    dome_light = sim_utils.DomeLightCfg(
        color=(0.9, 0.9, 0.9), intensity=500.0
    )


# =============================================================================
# 3. 観測設定
# =============================================================================
@configclass
class CartpolePIDObsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """
        エージェントの観測: [cart_x, cart_v, pole_rad, pole_omega]
        Isaac Lab の observation term として標準 mdp 関数を使用する。
        """
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# =============================================================================
# 4. アクション設定
# =============================================================================
@configclass
class CartpolePIDActionsCfg:
    """
    アクション設定: カスタム PID ActionTerm を登録する。

    Manager-based の ActionsCfg は「どんな ActionTerm を使うか」を宣言する。
    複数の ActionTerm を並べることも可能:
        joint_efforts  = mdp.JointEffortActionCfg(...)   # 標準
        pid_control    = CartpolePIDActionTermCfg(...)   # カスタム
    今回はカスタム PID ActionTerm のみ。
    """
    pid_control = CartpolePIDActionTermCfg(asset_name="robot", scale=0.0)


# =============================================================================
# 5. イベント設定（ランダムリセット）
# =============================================================================
@configclass
class CartpolePIDEventCfg:
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": sim_utils.SceneEntityCfg(
                "robot", joint_names=["slider_to_cart", "cart_to_pole"]
            ),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
    )


# =============================================================================
# 6. 環境設定
# =============================================================================
@configclass
class CartpolePIDManagerEnvCfg(ManagerBasedEnvCfg):
    """
    Manager-Based 環境設定。
    ManagerBasedEnv は報酬・終了条件を持たない「制御ループ」向けクラス。
    RL に使う場合は ManagerBasedRLEnvCfg を使用する。
    """
    scene:        CartpolePIDSceneCfg    = CartpolePIDSceneCfg(
                      num_envs=64, env_spacing=4.0
                  )
    observations: CartpolePIDObsCfg     = CartpolePIDObsCfg()
    actions:      CartpolePIDActionsCfg = CartpolePIDActionsCfg()
    events:       CartpolePIDEventCfg   = CartpolePIDEventCfg()

    def __post_init__(self):
        self.decimation = 2             # 制御 60Hz / 物理 120Hz
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation


# =============================================================================
# 7. メインループ
# =============================================================================
def main():
    env_cfg = CartpolePIDManagerEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = ManagerBasedEnv(cfg=env_cfg)

    # ゼロアクション（目標角=0 → 直立安定化 PID）
    actions = torch.zeros(env.num_envs, 1, device=env.device)

    obs, _ = env.reset()
    print(f"\n[INFO] Isaac Lab Manager PID 開始")
    print(f"  num_envs   = {env.num_envs}")
    print(f"  physics_dt = {env.cfg.sim.dt:.5f} s")
    print(f"  decimation = {env.cfg.decimation}")

    step = 0
    t_start = time.time()

    while simulation_app.is_running() and step < args_cli.max_steps:
        obs, _ = env.step(actions)
        step += 1

        if step % 60 == 0:
            t_sim = step * env.cfg.sim.dt * env.cfg.decimation
            # env_id=0 の観測から状態を取得
            robot = env.scene.articulations["robot"]
            p0 = robot.data.joint_pos[0, 1].item()  # pole angle
            x0 = robot.data.joint_pos[0, 0].item()  # cart position
            print(f"  ctrl={step:4d} t={t_sim:6.2f}s | "
                  f"θ={np.degrees(p0):+.3f}° x={x0:+.4f}m")

    elapsed = time.time() - t_start
    print(f"\n[INFO] 完了: {step} 制御ステップ / wall={elapsed:.1f}s")
    env.close()


if __name__ == "__main__":
    main()
