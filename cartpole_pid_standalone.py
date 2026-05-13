# =============================================================================
# cartpole_pid_standalone.py
# Isaac Sim 5.1.0 — CartPole PID Controller (Python Standalone)
# MATLAB比較用 CSV出力版
#
# 実行方法:
#   cd <isaac-sim-root>
#   ./python.sh /path/to/cartpole_pid_standalone.py
#   ./python.sh /path/to/cartpole_pid_standalone.py --headless
#   ./python.sh /path/to/cartpole_pid_standalone.py --headless --max_steps 7200
#
# 出力:
#   ./output/isaac_pid_log.csv  （MATLAB と列名・単位統一）
#
# =============================================================================
# 物理モデル（Isaac Sim cartpole.usd に合わせた値）
#   M = 1.0 kg  : カート質量
#   m = 0.1 kg  : ポール質量
#   l = 0.5 m   : ポール重心までの長さ
#   g = 9.81 m/s²
#   dt= 1/120 s : physics_dt
#
# PID 制御則:
#   F = Kp_pole*θ + Ki_pole*∫θdt + Kd_pole*θ̇   (ポール安定化)
#     + Kp_cart*x + Kd_cart*ẋ                    (カート位置回復)
#
# PIDゲイン（LQR由来: Q=diag([1,0.1,100,10]) R=0.001 → MATLAB と統一）
#   Kp_pole=497.62, Ki_pole=5.0, Kd_pole=141.36
#   Kp_cart=31.62,  Kd_cart=56.91
# =============================================================================

# ── [STEP 1] AppLauncher は全 isaacsim import より先に初期化する（必須）
import argparse
from isaacsim import AppLauncher

parser = argparse.ArgumentParser(description="CartPole PID Standalone")
parser.add_argument("--headless", action="store_true",
                    help="ヘッドレス実行（GUI なし）")
parser.add_argument("--max_steps", type=int, default=7200,
                    help="最大ステップ数 (7200 @ 120Hz = 60 s)")
parser.add_argument("--output_dir", type=str, default="./output")
parser.add_argument("--init_angle_deg", type=float, default=5.0,
                    help="ポール初期角度 [deg]")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── [STEP 2] Isaac Sim モジュールのインポート（AppLauncher の後）
import csv
import os
import time
import numpy as np

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage


# =============================================================================
# PID コントローラ
# =============================================================================
class CartPolePID:
    """
    CartPole 倒立振子 PID コントローラ

    制御則:
        up    = Kp_pole * θ
        ui    = Ki_pole * ∫θdt   (アンチワインドアップ: クランプ法)
        ud    = Kd_pole * θ̇
        u_pole = up + ui + ud

        u_cart = Kp_cart * x + Kd_cart * ẋ

        F = clip(u_pole + u_cart, -force_limit, +force_limit)  [N]

    ゲイン（LQR Q=diag([1,0.1,100,10]) R=0.001 から導出、MATLAB と統一）:
        Kp_pole=497.62, Ki_pole=5.0, Kd_pole=141.36
        Kp_cart=31.62,  Kd_cart=56.91
    """
    def __init__(self,
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
        self._integral = 0.0

    def reset(self) -> None:
        self._integral = 0.0

    def compute(self, cart_x: float, cart_v: float,
                theta: float, omega: float, dt: float):
        """
        Returns:
            force  : 制御力 [N]
            up, ui, ud : PID 各項 [N]  ← MATLABとの比較用
        """
        # 積分（アンチワインドアップ: クランプ法）
        self._integral = float(np.clip(
            self._integral + theta * dt,
            -self.integral_limit, self.integral_limit
        ))
        up = self.Kp_pole * theta
        ui = self.Ki_pole * self._integral
        ud = self.Kd_pole * omega
        u_pole = up + ui + ud
        u_cart = self.Kp_cart * cart_x + self.Kd_cart * cart_v
        force = float(np.clip(u_pole + u_cart,
                               -self.force_limit, self.force_limit))
        return force, up, ui, ud


# =============================================================================
# CSV ロガー（MATLAB との列名・単位を完全統一）
# =============================================================================
class CsvLogger:
    """
    Isaac Sim ログを CSV に保存する。

    列定義（cartpole_pid_matlab.m と完全一致）:
        time_s, cart_x_m, cart_v_ms, pole_rad, pole_deg,
        pole_omega_rads, force_N, pid_up, pid_ui, pid_ud

    バッファリング (flush_interval) で書き込み頻度を抑制。
    """
    HEADER = [
        "time_s", "cart_x_m", "cart_v_ms",
        "pole_rad", "pole_deg", "pole_omega_rads",
        "force_N", "pid_up", "pid_ui", "pid_ud",
    ]

    def __init__(self, filepath: str, flush_interval: int = 600):
        self._filepath = filepath
        self._flush_interval = flush_interval
        self._buf = []
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", newline="") as f:
            csv.writer(f).writerow(self.HEADER)

    def log(self, time_s, cart_x, cart_v, pole_rad, pole_omega,
            force, up, ui, ud):
        self._buf.append([
            f"{time_s:.6f}", f"{cart_x:.6f}", f"{cart_v:.6f}",
            f"{pole_rad:.6f}", f"{np.degrees(pole_rad):.4f}",
            f"{pole_omega:.6f}", f"{force:.4f}",
            f"{up:.4f}", f"{ui:.4f}", f"{ud:.4f}",
        ])
        if len(self._buf) >= self._flush_interval:
            self._flush()

    def _flush(self):
        with open(self._filepath, "a", newline="") as f:
            csv.writer(f).writerows(self._buf)
        self._buf.clear()

    def close(self):
        self._flush()
        print(f"[CSV] 保存完了: {os.path.abspath(self._filepath)}")


# =============================================================================
# シーン構築
# =============================================================================
def build_scene(world: World) -> Articulation:
    """
    Isaac Nucleus から CartPole USD をロードする。

    アセットパス:
        <nucleus_root>/Isaac/Robots/Classic/Cartpole/cartpole.usd
    Joint 0: slider_to_cart  (カート水平移動、制御入力: 力 F)
    Joint 1: cart_to_pole    (ポール回転、観測: 角度 θ)
    """
    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError(
            "Nucleus 未接続。Isaac Sim の Content Browser で "
            "/Isaac/ をマウントしてください。"
        )
    usd_path = assets_root + "/Isaac/Robots/Classic/Cartpole/cartpole.usd"
    print(f"[INFO] USD: {usd_path}")

    add_reference_to_stage(usd_path=usd_path, prim_path="/World/CartPole")

    from isaacsim.core.api.objects.ground_plane import GroundPlane
    GroundPlane(prim_path="/World/GroundPlane", z_position=0)

    cartpole = Articulation(prim_paths_expr="/World/CartPole", name="cartpole")
    world.scene.add(cartpole)
    return cartpole


# =============================================================================
# メインシミュレーションループ
# =============================================================================
def main():
    # ── ワールド生成
    # physics_dt = 1/120 s → MATLABの dt と統一
    world = World(
        physics_dt=1.0 / 120.0,
        rendering_dt=1.0 / 60.0,    # 描画は 2ステップに1回
        stage_units_in_meters=1.0,
    )
    cartpole = build_scene(world)
    world.reset()

    # dt の確認
    dt = world.get_physics_context().get_physics_dt()
    print(f"[INFO] physics_dt = {dt:.6f} s ({1/dt:.1f} Hz)")

    # ── PID・ロガー初期化
    pid = CartPolePID()
    csv_path = os.path.join(args.output_dir, "isaac_pid_log.csv")
    logger = CsvLogger(csv_path)

    # ── 初期状態（ポールを指定角度だけ傾ける）
    init_rad = np.radians(args.init_angle_deg)
    cartpole.set_joint_positions(np.array([0.0, init_rad]))
    cartpole.set_joint_velocities(np.zeros(2))
    world.step(render=False)   # バッファ反映

    print(f"[INFO] 初期角度 = {args.init_angle_deg}°  ({init_rad:.4f} rad)")
    print(f"[INFO] 最大ステップ = {args.max_steps}  ({args.max_steps * dt:.1f} s)\n")

    # ── メインループ
    step = 0
    sim_time = 0.0
    t_wall = time.time()

    while simulation_app.is_running() and step < args.max_steps:

        # 状態取得
        jpos = cartpole.get_joint_positions()   # [cart_x, pole_rad]
        jvel = cartpole.get_joint_velocities()  # [cart_v, pole_omega]
        cart_x    = float(jpos[0])
        pole_rad  = float(jpos[1])
        cart_v    = float(jvel[0])
        pole_omega = float(jvel[1])

        # PID 計算
        force, up, ui, ud = pid.compute(
            cart_x, cart_v, pole_rad, pole_omega, dt
        )

        # 力の印加（joint 0: slider_to_cart のみ）
        cartpole.set_joint_efforts(np.array([force, 0.0]))

        # ステップ前進
        world.step(render=not args.headless)
        step += 1
        sim_time += dt

        # ログ
        logger.log(sim_time, cart_x, cart_v, pole_rad, pole_omega,
                   force, up, ui, ud)

        # コンソール出力（毎秒 = 120ステップ）
        if step % 120 == 0:
            print(f"  t={sim_time:6.2f}s | "
                  f"x={cart_x:+.4f}m | "
                  f"θ={np.degrees(pole_rad):+.4f}° | "
                  f"F={force:+.2f}N")

        # 終了判定
        if abs(pole_rad) > np.pi / 2 or abs(cart_x) > 3.0:
            print(f"\n[WARN] 制御失敗 @ t={sim_time:.3f}s "
                  f"(θ={np.degrees(pole_rad):.1f}°, x={cart_x:.3f}m)")
            break

    elapsed = time.time() - t_wall
    logger.close()
    print(f"\n[INFO] 完了: {step} steps / sim={sim_time:.2f}s / "
          f"wall={elapsed:.1f}s")
    simulation_app.close()


if __name__ == "__main__":
    main()
