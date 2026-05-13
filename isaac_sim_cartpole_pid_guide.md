# Isaac Sim 5.1.0 — CartPole PID 制御 実施手順ガイド

> **目的**: Isaac Sim 5.1.0 における PID 制御の実装パターンを、3 つのワークフロー（GUI / Extension / Python Standalone）で明確化する。  
> **使用アセット**: NVIDIA 公式 CartPole サンプル（`CARTPOLE_CFG` / Isaac Nucleus asset）

---

## 0. 前提知識：Isaac Sim の 3 ワークフロー比較

| 項目 | GUI | Extension (Script Editor) | Python Standalone |
|---|---|---|---|
| 起動方法 | App Selector / `isaac-sim.sh` | Isaac Sim 内 `Window > Script Editor` | `./python.sh my_script.py` |
| Physics ステップ制御 | 自動（Play ボタン） | 非同期（コールバック登録） | 手動（`world.step()` を明示呼び出し） |
| ホットリロード | ― | ✅ ファイル保存で即反映 | ✗ 再起動が必要 |
| ヘッドレス実行 | ✗ | ✗ | ✅ `headless=True` |
| PID 用途 | プロトタイプ確認・可視化 | インタラクティブ開発・デバッグ | 大規模実験・CI・RL 連携 |

**研究開発での一般的な推奨フロー**:  
GUI でシーン確認 → Extension で PID コードを反復開発 → Standalone で実験データ収集

---

## 1. CartPole の構造と PID 設計方針

### 1-1. 関節構成（Isaac Sim 公式 USD）

```
/World/CartPole          ← ArticulationRoot
  └── cartbody           ← Cart（スライダ）
        └── pole         ← Pole（ヒンジ）

Joint 0: slider_to_cart  → 制御入力: 水平力 F [N]（effort target）
Joint 1: cart_to_pole    → 観測: ポール角度 θ [rad]
```

### 1-2. 状態ベクトルと PID 設計

| 変数 | 取得 API | 役割 |
|---|---|---|
| `cart_x` | `joint_pos[:,0]` | カート位置 [m] |
| `cart_v` | `joint_vel[:,0]` | カート速度 [m/s] |
| `pole_θ` | `joint_pos[:,1]` | ポール角度 [rad]（目標: 0） |
| `pole_ω` | `joint_vel[:,1]` | ポール角速度 [rad/s] |

**PID 構造（倒立振子の標準構成）**:

```
u = Kp_pole × θ  +  Kd_pole × ω  +  Ki_pole × ∫θdt   (ポール安定化項)
  + Kp_cart × x  +  Kd_cart × v                        (カート位置回復項)
```

Cart への力 `u [N]` を `set_joint_effort_target()` で印加する。

---

## 2. ワークフロー 1：GUI

GUI ワークフローは **コードなしで CartPole の動作確認** を行い、PID パラメータの直感的な感覚を掴む用途に最適。

### 手順

#### Step 1. Isaac Sim 起動・シーン作成

1. `isaac-sim.sh`（Linux）または App Selector から Isaac Sim を起動
2. `File > New` で空ステージを作成

#### Step 2. CartPole アセットをステージに追加

1. メニューバー `Window > Examples > Robotics Examples` を開く
2. `CORE API > Cartpole` を選択して `LOAD` ボタンを押す  
   → `/World/CartPole` prim が追加される

   **または**、Content Browser から直接ドラッグ：
   ```
   /Isaac/Robots/Classic/Cartpole/cartpole.usd
   ```

#### Step 3. Joint Drive の確認（GUIのPID相当設定）

1. Stage パネルで `/World/CartPole/cartbody/slider_to_cart` を選択
2. Property パネルで `Physics > Drive > Linear` を確認
   - `Stiffness`（= Kp 相当）: `0`（force control のため 0 のまま）
   - `Damping`（= Kd 相当）: `0`
3. アクチュエータタイプが **Force** モードであることを確認

#### Step 4. シミュレーション実行と観察

1. `Play` ボタンでシミュレーション開始
2. ポールが倒れていく様子を確認（制御なし）
3. `Pause` → `Reset` でリセット

**GUI でできる PID 的調整**:  
Joint Drive の `Stiffness` / `Damping` をインタラクティブに変更することで、  
位置制御モード（`Position` Drive）の Kp/Kd を GUI 上から調整可能。  
ただし、倒立振子の「力制御 PID」は Script Editor か Standalone で実装する。

---

## 3. ワークフロー 2：Extension（Script Editor）

Script Editor を使った Extension ワークフローは、**ホットリロードでの反復開発**に最適。  
シミュレーションを止めずにPIDゲインを変更・確認できる。

### 手順

#### Step 1. Isaac Sim 起動・CartPole シーン準備

```
Isaac Sim 起動 → File > New → Window > Examples > Robotics Examples
→ CORE API > Cartpole > LOAD
```

#### Step 2. Script Editor を開く

`Window > Script Editor` → 新しいタブを開く

#### Step 3. PID コードを Script Editor に貼り付けて実行

以下を **まるごとコピー**して Script Editor に貼り付け、`Run` ボタンを押す。

```python
# ==============================================================
# Isaac Sim 5.1.0 — CartPole PID Controller (Extension/Script Editor)
# 対象: Window > Examples > Robotics Examples > CORE API > Cartpole でロードした
#       /World/CartPole アセット
# 実行: Script Editor の Run ボタン（Playボタンを押す前に実行）
# ==============================================================
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation

# ── PID コントローラクラス ──────────────────────────────────────
class CartPolePID:
    """
    倒立振子 PID コントローラ
    制御則: u = Kp_pole*θ + Kd_pole*ω + Ki_pole*∫θdt + Kp_cart*x + Kd_cart*v
    出力: カートへの水平力 [N]
    """
    def __init__(self,
                 Kp_pole=3000.0, Kd_pole=800.0, Ki_pole=50.0,
                 Kp_cart=200.0,  Kd_cart=100.0,
                 integral_limit=10.0):
        self.Kp_pole = Kp_pole
        self.Kd_pole = Kd_pole
        self.Ki_pole = Ki_pole
        self.Kp_cart = Kp_cart
        self.Kd_cart = Kd_cart
        self.integral_limit = integral_limit
        self._integral = 0.0

    def reset(self):
        self._integral = 0.0

    def compute(self, cart_x, cart_v, pole_theta, pole_omega, dt):
        # 積分項（アンチワインドアップ付き）
        self._integral += pole_theta * dt
        self._integral = np.clip(self._integral,
                                 -self.integral_limit, self.integral_limit)

        # ポール安定化 + カート位置回復
        u = (self.Kp_pole * pole_theta
           + self.Kd_pole * pole_omega
           + self.Ki_pole * self._integral
           + self.Kp_cart * cart_x
           + self.Kd_cart * cart_v)
        return float(u)


# ── グローバル変数（コールバック間で共有） ─────────────────────
_pid = CartPolePID()
_prev_time = None
_step_count = 0

# ── メイン初期化 ───────────────────────────────────────────────
world = World.instance()
if world is None:
    world = World(stage_units_in_meters=1.0)

# CartPole ArticulationをCore APIで取得
cartpole = Articulation(prim_paths_expr="/World/CartPole")
world.scene.add(cartpole)

# ── Physics コールバック登録 ───────────────────────────────────
def pid_step(step_size: float):
    """毎 physics ステップで呼ばれる PID 制御コールバック"""
    global _prev_time, _step_count, _pid

    _step_count += 1
    dt = step_size  # [sec]

    # 状態取得
    # joint 0: slider_to_cart (カート位置/速度)
    # joint 1: cart_to_pole   (ポール角度/角速度)
    joint_pos = cartpole.get_joint_positions()   # shape: (2,)
    joint_vel = cartpole.get_joint_velocities()  # shape: (2,)

    if joint_pos is None or joint_vel is None:
        return

    cart_x    = float(joint_pos[0])
    pole_theta = float(joint_pos[1])
    cart_v    = float(joint_vel[0])
    pole_omega = float(joint_vel[1])

    # PID 計算
    force = _pid.compute(cart_x, cart_v, pole_theta, pole_omega, dt)
    force = np.clip(force, -5000.0, 5000.0)  # 飽和制限 [N]

    # カートに力を印加（joint 0 = slider_to_cart）
    # effort target: shape (num_joints,) → カートのみ制御
    efforts = np.array([force, 0.0])
    cartpole.set_joint_efforts(efforts)

    # ログ（100ステップごと）
    if _step_count % 100 == 0:
        print(f"[PID] step={_step_count:5d} | "
              f"cart_x={cart_x:+.3f}m | "
              f"pole_θ={np.degrees(pole_theta):+.2f}° | "
              f"force={force:+.1f}N")

# コールバック登録（重複登録防止）
try:
    world.remove_physics_callback("cartpole_pid")
except Exception:
    pass
world.add_physics_callback("cartpole_pid", pid_step)

print("=" * 50)
print("CartPole PID コントローラ 登録完了")
print("Play ボタンを押してシミュレーション開始")
print("Pause → Reset でリセット")
print("=" * 50)
```

#### Step 4. シミュレーション開始・確認

1. `Play` ボタンを押す
2. Script Editor 下部のコンソールにログが流れることを確認
3. ポールが立った状態で安定することを確認

#### Step 5. ゲインのホットリロード調整

別のタブで以下を実行するか、元のスクリプトの `Kp_pole` 等を変更して `Run` し直す：

```python
# ゲインのみ変更（シミュレーション継続中に実行可能）
import builtins
# グローバルの _pid に直接アクセス
import __main__
__main__._pid.Kp_pole = 5000.0
__main__._pid.Kd_pole = 1200.0
print("ゲイン更新完了")
```

#### Step 6. リセット処理

```python
# Script Editor で実行 — PID 積分項のリセット
__main__._pid.reset()
__main__._step_count = 0
print("PID リセット完了")
```

---

## 4. ワークフロー 3：Python Standalone

Standalone ワークフローは **物理・レンダリングステップを完全に手動制御** できる。  
研究開発での標準実装パターン（RL 連携・データ収集・ヘッドレス実行）に対応。

### ファイル構成

```
<isaac-sim-root>/
├── python.sh                        ← Isaac Sim 専用 Python インタープリタ
└── standalone_examples/
    └── pid_cartpole/
        └── cartpole_pid_standalone.py   ← 本スクリプト
```

### スクリプト全文

以下を `cartpole_pid_standalone.py` として保存：

```python
# ==============================================================
# Isaac Sim 5.1.0 — CartPole PID Controller (Python Standalone)
#
# 実行方法:
#   cd <isaac-sim-root>
#   ./python.sh standalone_examples/pid_cartpole/cartpole_pid_standalone.py
#
# ヘッドレス実行:
#   ./python.sh cartpole_pid_standalone.py --headless
#
# 研究開発パターン:
#   - AppLauncher で SimulationApp を起動（必須: import前に初期化）
#   - World + Articulation で環境構築
#   - world.step(render=True) で物理・描画を同期ステップ
#   - データをログ・csvに保存
# ==============================================================

# ── [STEP 1] AppLauncher は必ず最初に初期化 ─────────────────────
import argparse
from isaacsim import AppLauncher

parser = argparse.ArgumentParser(description="CartPole PID Standalone")
parser.add_argument("--headless", action="store_true",
                    help="ヘッドレス実行（GUI なし）")
parser.add_argument("--max_steps", type=int, default=3000,
                    help="最大シミュレーションステップ数")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# AppLauncher 初期化（これより前に isaacsim モジュールをインポートしない）
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── [STEP 2] Isaac Sim モジュールのインポート ──────────────────
import numpy as np
import csv
import os
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage

# ── [STEP 3] PID コントローラ定義 ───────────────────────────────
class CartPolePID:
    """
    CartPole 倒立振子 PID コントローラ
    
    制御則:
        u = Kp_θ·θ + Kd_θ·θ̇ + Ki_θ·∫θdt   (ポール角度フィードバック)
          + Kp_x·x + Kd_x·ẋ                 (カート位置回復)
    
    Args:
        Kp_pole: ポール角度比例ゲイン
        Kd_pole: ポール角速度微分ゲイン
        Ki_pole: ポール角度積分ゲイン（ドリフト補正）
        Kp_cart: カート位置比例ゲイン
        Kd_cart: カート速度微分ゲイン
        integral_limit: 積分ワインドアップ防止リミット
        force_limit: 出力飽和 [N]
    """
    def __init__(self,
                 Kp_pole: float = 3000.0,
                 Kd_pole: float = 800.0,
                 Ki_pole: float = 50.0,
                 Kp_cart: float = 200.0,
                 Kd_cart: float = 100.0,
                 integral_limit: float = 10.0,
                 force_limit: float = 5000.0):
        self.Kp_pole = Kp_pole
        self.Kd_pole = Kd_pole
        self.Ki_pole = Ki_pole
        self.Kp_cart = Kp_cart
        self.Kd_cart = Kd_cart
        self.integral_limit = integral_limit
        self.force_limit = force_limit
        self._integral = 0.0

    def reset(self) -> None:
        """積分項をリセット（エピソード開始時に呼ぶ）"""
        self._integral = 0.0

    def compute(self,
                cart_x: float,
                cart_v: float,
                pole_theta: float,
                pole_omega: float,
                dt: float) -> float:
        """
        PID 制御力を計算して返す
        
        Args:
            cart_x:     カート位置 [m]
            cart_v:     カート速度 [m/s]
            pole_theta: ポール角度 [rad]（直立 = 0）
            pole_omega: ポール角速度 [rad/s]
            dt:         制御周期 [sec]
        
        Returns:
            force: カートへの制御力 [N]
        """
        # 積分項（アンチワインドアップ）
        self._integral += pole_theta * dt
        self._integral = np.clip(
            self._integral, -self.integral_limit, self.integral_limit
        )

        # PID 計算
        u_pole = (self.Kp_pole * pole_theta
                + self.Kd_pole * pole_omega
                + self.Ki_pole * self._integral)
        u_cart = (self.Kp_cart * cart_x
                + self.Kd_cart * cart_v)
        force = u_pole + u_cart

        # 飽和制限
        return float(np.clip(force, -self.force_limit, self.force_limit))


# ── [STEP 4] ワールド・シーン構築 ──────────────────────────────
def build_scene(world: World) -> Articulation:
    """
    Isaac Nucleus から CartPole USD をロードしてシーンを構築する。
    
    アセットパス例:
        /Isaac/Robots/Classic/Cartpole/cartpole.usd
    """
    assets_root = get_assets_root_path()
    if assets_root is None:
        # ローカルフォールバック（Nucleus 未接続環境）
        raise RuntimeError(
            "Nucleus サーバーに接続できません。"
            "Isaac Sim の Content ブラウザで Nucleus 接続を確認してください。"
        )

    cartpole_usd = assets_root + "/Isaac/Robots/Classic/Cartpole/cartpole.usd"
    print(f"[INFO] CartPole USD: {cartpole_usd}")

    # ステージにアセットを追加
    add_reference_to_stage(usd_path=cartpole_usd, prim_path="/World/CartPole")

    # Ground Plane
    from isaacsim.core.api.objects.ground_plane import GroundPlane
    GroundPlane(prim_path="/World/GroundPlane", z_position=0)

    # Articulation オブジェクト生成
    cartpole = Articulation(prim_paths_expr="/World/CartPole",
                            name="cartpole")
    world.scene.add(cartpole)
    return cartpole


# ── [STEP 5] データロガー ────────────────────────────────────────
class DataLogger:
    """シミュレーションデータを CSV に保存するロガー"""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._rows = []
        self._headers = ["step", "time_s",
                          "cart_x", "cart_v",
                          "pole_deg", "pole_omega",
                          "force_N"]

    def log(self, step, t, cart_x, cart_v, pole_theta, pole_omega, force):
        self._rows.append([
            step, round(t, 4),
            round(cart_x, 4), round(cart_v, 4),
            round(np.degrees(pole_theta), 3), round(pole_omega, 4),
            round(force, 2)
        ])

    def save(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._headers)
            writer.writerows(self._rows)
        print(f"[INFO] ログ保存: {self.filepath}")


# ── [STEP 6] メインシミュレーションループ ───────────────────────
def main():
    # ワールド生成（physics dt = 1/120 sec = 8.33 ms）
    world = World(
        physics_dt=1.0 / 120.0,
        rendering_dt=1.0 / 60.0,   # 描画は 60fps（2ステップに1回）
        stage_units_in_meters=1.0
    )

    # シーン構築
    cartpole = build_scene(world)

    # ワールドのリセット（初期化・バッファ確保）
    world.reset()

    # PID コントローラ
    pid = CartPolePID(
        Kp_pole=3000.0, Kd_pole=800.0, Ki_pole=50.0,
        Kp_cart=200.0,  Kd_cart=100.0,
    )

    # データロガー
    logger = DataLogger("/tmp/isaac_cartpole_pid_log.csv")

    # 物理タイムステップ（dt）を取得
    physics_dt = world.get_physics_context().get_physics_dt()
    print(f"[INFO] physics_dt = {physics_dt:.5f} sec "
          f"({1/physics_dt:.1f} Hz)")

    # ────────────────────────────────────────────────────────
    # メインループ
    # ────────────────────────────────────────────────────────
    print("\n[INFO] シミュレーション開始...")
    step = 0
    sim_time = 0.0
    is_done = False

    while simulation_app.is_running() and step < args.max_steps:

        # ── エピソードリセット判定 ──
        if step == 0 or is_done:
            world.reset()
            pid.reset()
            is_done = False
            print(f"[INFO] エピソードリセット (step={step})")

        # ── 状態取得 ──
        joint_pos = cartpole.get_joint_positions()
        joint_vel = cartpole.get_joint_velocities()

        cart_x    = float(joint_pos[0])   # [m]
        pole_theta = float(joint_pos[1])  # [rad]
        cart_v    = float(joint_vel[0])   # [m/s]
        pole_omega = float(joint_vel[1])  # [rad/s]

        # ── PID 制御計算 ──
        force = pid.compute(cart_x, cart_v, pole_theta, pole_omega,
                            physics_dt)

        # ── アクション印加: joint 0 (cart) のみ effort ──
        efforts = np.array([force, 0.0])
        cartpole.set_joint_efforts(efforts)

        # ── シミュレーション 1 ステップ前進 ──
        world.step(render=not args.headless)
        step += 1
        sim_time += physics_dt

        # ── ログ記録 ──
        logger.log(step, sim_time, cart_x, cart_v,
                   pole_theta, pole_omega, force)

        # ── コンソール出力（200 ステップごと） ──
        if step % 200 == 0:
            print(f"  step={step:4d} | t={sim_time:.2f}s | "
                  f"cart={cart_x:+.3f}m | "
                  f"pole={np.degrees(pole_theta):+.2f}° | "
                  f"F={force:+.1f}N")

        # ── 終了判定（ポール倒れ / カートはみ出し） ──
        if (abs(pole_theta) > np.pi / 2 or abs(cart_x) > 3.0):
            print(f"[WARN] 制御失敗 step={step} "
                  f"(θ={np.degrees(pole_theta):.1f}°, x={cart_x:.2f}m)")
            is_done = True

    # ── 後処理 ──
    logger.save()
    print(f"\n[INFO] 完了: {step} steps / {sim_time:.2f} sec")
    simulation_app.close()


if __name__ == "__main__":
    main()
```

### 実行コマンド

```bash
cd /path/to/isaac-sim-5.1.0

# 通常実行（GUI 表示あり）
./python.sh standalone_examples/pid_cartpole/cartpole_pid_standalone.py

# ヘッドレス実行（CI・大規模実験）
./python.sh standalone_examples/pid_cartpole/cartpole_pid_standalone.py \
    --headless --max_steps 6000

# GPU 指定実行
./python.sh standalone_examples/pid_cartpole/cartpole_pid_standalone.py \
    --gpu-physics-device 0 --headless
```

---

## 5. PID 実装の核心：3 ワークフローの API 対応表

| 操作 | Extension (Script Editor) | Python Standalone |
|---|---|---|
| World 取得 | `World.instance()` | `World(physics_dt=...)` |
| Articulation 取得 | `Articulation(prim_paths_expr=...)` | 同左 |
| joint 位置読み取り | `cartpole.get_joint_positions()` | 同左 |
| joint 速度読み取り | `cartpole.get_joint_velocities()` | 同左 |
| 力の印加 | `cartpole.set_joint_efforts(efforts)` | 同左 |
| ステップ進行 | Physics コールバック（非同期） | `world.step(render=True)` |
| リセット | `world.reset()` | `world.reset()` |
| ゲイン変更 | スクリプト再 Run / 変数直接書き換え | スクリプト再起動 |

---

## 6. PID チューニングの指針（研究開発標準）

### 6-1. Ziegler-Nichols 法（出発点）

1. `Ki_pole = 0, Kd_pole = 0` でまず `Kp_pole` のみ上げる
2. 振動が始まる限界ゲイン `Ku` と振動周期 `Tu` を記録
3. PID ゲインの初期値:

| パラメータ | 式 |
|---|---|
| `Kp_pole` | `0.6 × Ku` |
| `Ki_pole` | `2 × Kp / Tu` |
| `Kd_pole` | `Kp × Tu / 8` |

### 6-2. Isaac Sim での実用値（参考）

```python
# ポール安定化（高ゲイン・高周波応答）
Kp_pole = 3000.0   # ~ 5000.0
Kd_pole = 800.0    # ~ 1200.0
Ki_pole = 50.0     # 積分は小さく（ワインドアップ対策）

# カート位置回復（低ゲイン・ゆっくり）
Kp_cart = 200.0    # ~ 500.0
Kd_cart = 100.0    # ~ 300.0

# アンチワインドアップ
integral_limit = 10.0
force_limit    = 5000.0  # [N]
```

> **ゲインが大きすぎると**：力の飽和 → 振動発散  
> **ゲインが小さすぎると**：ポール倒れへの反応遅れ  
> **Ki が大きすぎると**：積分ワインドアップ → 大振動

---

## 7. トラブルシューティング

| 症状 | 原因 | 対処 |
|---|---|---|
| `joint_pos` が `None` を返す | Articulation が物理初期化前 | `world.reset()` 後に取得する |
| CartPole USD が見つからない | Nucleus 未接続 | Content Browser で `/Isaac/` をマウント |
| ポールがすぐ倒れる | ゲイン不足 / dt ズレ | `Kp_pole` を 2 倍にして試す |
| `set_joint_efforts` が効かない | Joint Drive が Position モード | USD で Drive type を `None` に変更 |
| Script Editor でエラー | Articulation がシーンにない | LOAD を先に実行してから Run |
| Standalone でモジュールが見つからない | システム Python で実行している | `./python.sh` を使う（Isaac Sim 専用 Python）|

---

## 8. 次のステップ

- **LQR への発展**: 線形化した状態方程式を `scipy.linalg.solve_discrete_are` で解いて最適ゲインを求める
- **Isaac Lab との連携**: `cartpole_env.py` の `DirectRLEnv` を継承して PID ベースラインと RL ポリシーを比較実験
- **複数環境**: Standalone の `Articulation` を `prim_paths_expr="/World/CartPole_*"` でベクトル化し並列制御

---

*Isaac Sim 5.1.0 / Isaac Lab 対応 | 参考: NVIDIA 公式ドキュメント・IsaacLab GitHub・研究コミュニティ実装*
