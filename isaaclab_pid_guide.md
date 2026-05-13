# Isaac Lab — CartPole PID 制御 検証ガイド

> **対象**: Isaac Lab (on Isaac Sim 5.1.0)  
> **目的**: Isaac Lab の 2 ワークフロー（Direct / Manager-Based）における  
>           PID 制御の実装位置と役割を明確化する

---

## 0. Isaac Lab とは何か

Isaac Sim と Isaac Lab の関係を整理すると:

```
Isaac Sim (シミュレーター本体)
  └── Isaac Lab (RL/制御タスク フレームワーク)
        ├── DirectRLEnv         ← RL タスクを直接実装（自由度高）
        ├── ManagerBasedEnv     ← 制御ループ（報酬・終了条件なし）
        └── ManagerBasedRLEnv   ← RL タスクをマネージャで宣言的に実装
```

Isaac Lab を使う主目的は **「大規模並列RL学習」** だが、
PID・LQR などの古典制御を実装してベースライン化することも研究開発の標準パターン。

---

## 1. Isaac Lab の 2 ワークフローと PID の位置

### 1-1. Direct Workflow (`DirectRLEnv`)

```
env.step(action)
  │
  ├─ _pre_physics_step(action)   ← ★ここで PID を計算
  │     PID(state) → force
  │
  ├─ _apply_action()             ← ★ここで力を印加
  │     robot.set_joint_effort_target(force)
  │     （decimation 回だけ繰り返される）
  │
  ├─ sim.step()  × decimation    ← PhysX 物理演算
  │
  ├─ _get_observations()         ← 観測ベクトル更新
  ├─ _get_rewards()              ← 報酬計算
  └─ _get_dones()                ← エピソード終了判定
```

**PID の実装位置**: `_pre_physics_step()` + `_apply_action()`

| 関数 | 呼ばれる周期 | PID での役割 |
|---|---|---|
| `_pre_physics_step()` | 制御ステップ (60Hz) | PID 計算、力バッファ更新 |
| `_apply_action()` | 物理ステップ (120Hz) | 力の印加（decimation=2 なら 2回） |

### 1-2. Manager-Based Workflow (`ManagerBasedEnv`)

```
env.step(action)
  │
  ├─ ActionManager.process_actions(action)
  │     └─ CartpolePIDActionTerm.process_actions()  ← 目標値受け取り
  │
  ├─ sim.step() × decimation
  │     各物理ステップで:
  │     └─ CartpolePIDActionTerm.apply_actions()    ← ★PID演算+力印加
  │
  └─ ObservationManager.compute()  ← 観測更新
```

**PID の実装位置**: カスタム `ActionTerm` クラスの `apply_actions()`

---

## 2. ワークフロー別 実装ファイル一覧

| ファイル | ワークフロー | PID実装位置 | 用途 |
|---|---|---|---|
| `cartpole_pid_direct.py` | Direct | `_pre_physics_step` | 並列PID検証・CSV出力 |
| `cartpole_pid_manager.py` | Manager-Based | カスタム `ActionTerm` | 再利用可能な制御モジュール |

---

## 3. Isaac Lab での PID — 4 つの活用パターン

### パターン① 固定ゲイン PID（ベースライン）
*今回の実装*

```python
# _pre_physics_step / apply_actions() 内
force = Kp*θ + Ki*∫θdt + Kd*θ̇ + Kp_cart*x + Kd_cart*v
robot.set_joint_effort_target(force)
```

`action` は無視。純粋な古典制御として Isaac Lab の並列環境で動かす。  
→ **RL 学習前のベースライン評価**に使用。

---

### パターン② PID の目標値を RL が出力する（階層制御）

```python
# process_actions(action):
theta_ref = action * scale      # RL が目標角度を出力

# apply_actions():
error = theta_ref - theta       # PID が誤差を追従
force = Kp * error + Kd * (-omega)
```

RL の action 空間 = `[-1, 1]` → 目標ポール角 `[-scale, scale] rad`  
→ RL は「どちらに傾けるか」だけ決め、細かい安定化は PID が行う。  
→ **安全制約付き RL** や **Sim-to-Real** で有効。

---

### パターン③ PID ゲインを RL が適応する（アダプティブ制御）

```python
# action = [delta_Kp, delta_Kd]  (ゲインの補正量)
# apply_actions():
Kp_eff = self.Kp_base + action[0] * gain_scale
Kd_eff = self.Kd_base + action[1] * gain_scale
force = Kp_eff * theta + Kd_eff * omega
```

→ 事前設計した PID を RL で微調整する研究パターン。

---

### パターン④ PID ゲインを自動識別（物理パラメータ不明時）

```python
# Isaac Lab の Domain Randomization でポール質量・長さをランダム化
EventCfg:
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        params={"mass_distribution_params": (0.05, 0.2)}
    )

# PID はロバストゲインで固定し、RL でモデル誤差を補償する action を学習
```

→ 質量・長さが変わっても倒れない Policy の学習（Sim-to-Real 標準手法）。

---

## 4. `env.step()` の内部フロー詳細（decimation=2 の場合）

```
env.step(action)                                    # 制御ステップ: 60Hz
│
├── [1] ActionManager.process_actions(action)       # 60Hz
│       └── CartpolePIDActionTerm.process_actions() # 目標値更新
│
├── ── 物理ループ (2回) ──────────────────────────── # 120Hz × 2
│   ┌─ loop iter 1
│   │    ActionTerm.apply_actions()   ← PID演算 + force 印加
│   │    SimulationContext.step()     ← PhysX 物理演算
│   └─ loop iter 2
│        ActionTerm.apply_actions()   ← 同じ theta_ref で再計算
│        SimulationContext.step()
│
├── [2] ObservationManager.compute()   # 60Hz: 観測更新
└── [3] (RLEnv の場合) reward, done    # 60Hz: 報酬・終了判定
```

**重要**: `apply_actions()` は毎物理ステップ(120Hz)で呼ばれる。  
→ PID の積分・微分は 120Hz で更新される（dt = 1/120 s）。  
→ `process_actions()` の theta_ref は 60Hz で更新される。

---

## 5. Isaac Sim Standalone との比較

| 項目 | Isaac Sim Standalone | Isaac Lab Direct | Isaac Lab Manager |
|---|---|---|---|
| PID 実装 | `world.step()` のループ内 | `_pre_physics_step()` | `ActionTerm.apply_actions()` |
| 並列環境 | 基本 1環境 | num_envs 並列 | num_envs 並列 |
| 状態取得 | `get_joint_positions()` | `robot.data.joint_pos` | 同左 |
| 力の印加 | `set_joint_efforts()` | `set_joint_effort_target()` | 同左 |
| 制御周期 | physics_dt = 1/120 s | decimation × sim.dt | 同左 |
| テンソル | NumPy | **torch.Tensor (GPU)** | torch.Tensor (GPU) |
| RL連携 | 手動実装 | **直接連携可** | 直接連携可 |
| CSV出力 | 簡単 | `env.step()` 後に取得 | 同左 |

---

## 6. 実行コマンド

### Direct Workflow（PID ベースライン）

```bash
cd <IsaacLab-root>

# 64並列環境でPID実行
./isaaclab.sh -p /path/to/cartpole_pid_direct.py --num_envs 64

# 1環境 + CSV出力（MATLABとの比較用）
./isaaclab.sh -p /path/to/cartpole_pid_direct.py \
    --num_envs 1 \
    --max_steps 7200 \
    --csv_out ./output/isaaclab_direct_pid.csv

# ヘッドレス（高速）
./isaaclab.sh -p /path/to/cartpole_pid_direct.py \
    --num_envs 1024 --headless
```

### Manager-Based Workflow（カスタム ActionTerm）

```bash
./isaaclab.sh -p /path/to/cartpole_pid_manager.py --num_envs 64
```

---

## 7. CSV 出力列（3ツール統一）

```
time_s, cart_x_m, cart_v_ms, pole_rad, pole_deg, pole_omega_rads, force_N, pid_up, pid_ui, pid_ud
```

| ツール | 出力ファイル |
|---|---|
| Isaac Sim Standalone | `./output/isaac_pid_log.csv` |
| Isaac Lab Direct | `./output/isaaclab_direct_pid.csv` |
| MATLAB | `./output/matlab_pid_log.csv` |

比較: `compare_isaac_matlab.m` の入力を `isaaclab_direct_pid.csv` に変えるだけで  
Isaac Lab Direct の結果を MATLAB と比較できる。

---

## 8. よくある疑問

**Q. `_apply_action()` が decimation 回呼ばれるのに積分が 2 倍になる？**  
A. ならない。`apply_actions()` 内で積分を計算する場合、dt = physics_dt (1/120s) を使う。  
decimation=2 なら 2回 × (1/120s) = 1/60s の積分になるが、これは正しい制御周期。  
ただし `process_actions()` で theta_ref を更新するなら積分は `apply_actions()` で行うこと。

**Q. Isaac Lab の環境を Gymnasium として使うには？**  
```python
import gymnasium as gym
env = gym.make("Isaac-Cartpole-Direct-v0",
               cfg=CartpolePIDEnvCfg(),
               render_mode="human")
obs, _ = env.reset()
for _ in range(1000):
    action = my_pid_action(obs)      # PID出力
    obs, reward, done, trunc, info = env.step(action)
```

**Q. RL 学習に切り替えるには？**  
`CartpolePIDEnv(DirectRLEnv)` をそのまま RL ライブラリに渡せる:
```bash
# rsl_rl での学習（標準 CartPole タスクを置き換える場合）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Cartpole-Direct-v0
```
PID をベースラインとして `--checkpoint` 比較するのが研究開発の標準フロー。

---

*Isaac Lab 公式ドキュメント / IsaacLab GitHub に基づく (2025年)*
