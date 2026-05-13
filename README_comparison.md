# Isaac Sim vs MATLAB — CartPole PID 比較ガイド

## ファイル一覧

| ファイル | 説明 |
|---|---|
| `cartpole_pid_standalone.py` | Isaac Sim 5.1.0 Standalone スクリプト（CSV出力付き） |
| `cartpole_pid_matlab.m` | MATLAB 実装（完全非線形モデル + RK4 + CSV出力） |
| `compare_isaac_matlab.m` | 両CSVを読み込んで比較プロットを生成 |
| `matlab_pid_log_sample.csv` | MATLAB モデルのサンプル出力（Python検証版） |

---

## 実行手順

### Step 1: MATLAB でシミュレーション

```matlab
>> cartpole_pid_matlab          % CSV + プロット生成
% → ./output/matlab_pid_log.csv
% → ./output/matlab_pid_plot.png
```

### Step 2: Isaac Sim で実行

```bash
cd <isaac-sim-root>
./python.sh /path/to/cartpole_pid_standalone.py --headless --max_steps 7200
# → ./output/isaac_pid_log.csv
```

### Step 3: MATLAB で比較プロット

```matlab
>> compare_isaac_matlab
% → ./output/comparison_plot.png + コンソールに RMSE サマリー
```

---

## CSV 列定義（両ツール共通）

| 列名 | 単位 | 説明 |
|---|---|---|
| `time_s` | s | シミュレーション経過時間 |
| `cart_x_m` | m | カート位置 |
| `cart_v_ms` | m/s | カート速度 |
| `pole_rad` | rad | ポール角度（直立=0） |
| `pole_deg` | deg | ポール角度（可読性のため併記） |
| `pole_omega_rads` | rad/s | ポール角速度 |
| `force_N` | N | 制御力（カートへの水平力） |
| `pid_up` | N | PID P項（比例）値 |
| `pid_ui` | N | PID I項（積分）値 |
| `pid_ud` | N | PID D項（微分）値 |

---

## 物理モデルの対応関係

### Isaac Sim (PhysX 物理エンジン)

- USD ファイルから質量・慣性を**ジオメトリの体積と密度から自動計算**
- 実際の値はスクリプト内で `cartpole.get_body_masses()` で確認可能
- 積分器: PhysX 独自の implicit Euler 系

### MATLAB (解析的非線形モデル)

- 質量・長さを**手動で設定**（Isaac Sim USD と合わせた推定値）
- 積分器: RK4（4次ルンゲクッタ）
- 運動方程式: Florian 2007 の完全非線形モデル

### 差異の原因と解釈

```
差異発生要因（大→小の順）:
  1. 積分器の差: PhysX(implicit Euler) vs RK4 → 短時間の軌道差
  2. 物理パラメータの差: USD自動計算 vs 手動設定 → 定常ゲイン差
  3. 数値精度: float32(PhysX) vs float64(MATLAB) → 長時間に蓄積

実験的確認:
  - RMSE が 0.01° 以下なら物理モデルの一致度が高い
  - 両者が同じ定常状態に収束すれば PID ゲインは適切
  - 力の波形形状が一致していれば制御則は同一
```

---

## PID ゲイン設計根拠

### LQR 設計（線形化モデル）

状態空間線形化（θ≈0 近傍）:

```
A = [[0, 1,    0,    0   ],
     [0, 0, -0.72,   0   ],
     [0, 0,    0,    1   ],
     [0, 0, 15.79,   0   ]]

B = [[0], [0.98], [0], [-1.46]]

Q = diag([1.0, 0.1, 100.0, 10.0])  # 位置, 速度, 角度, 角速度
R = [[0.001]]                        # 制御エネルギー
```

LQR 解: `K = [-31.6, -56.9, -497.6, -141.4]` (x, ẋ, θ, θ̇)

### PID へのマッピング

```python
# LQR K → PID ゲイン（符号を正に調整）
Kp_pole = |K[2]| = 497.62   # θ 比例ゲイン
Kd_pole = |K[3]| = 141.36   # θ̇ 微分ゲイン
Kp_cart = |K[0]| = 31.62    # x 比例ゲイン
Kd_cart = |K[1]| = 56.91    # ẋ 微分ゲイン
Ki_pole = 5.0                # 積分（LQR非対応項、実験的に設定）
```

### 力の飽和・積分ワインドアップ対策

```
force_limit    = 500.0 N   # LQR の実動作範囲（~43N）に十分な余裕
integral_limit = 0.5       # ポール角度の積分値 [rad·s]（ドリフト抑制）
```

---

## サンプル出力確認

MATLAB モデル（Python再現版）の先頭3行:

```csv
time_s,cart_x_m,cart_v_ms,pole_rad,pole_deg,pole_omega_rads,force_N,pid_up,pid_ui,pid_ud
0.000000,0.000000,0.000000,0.087266,5.0000,0.000000,43.4292,43.4292,0.0000,0.0000
0.008333,0.001468,0.352378,0.085117,4.8768,-0.515988,-10.4769,42.3619,0.0036,-52.8424
```

60秒後の定常状態:
```
θ ≈ 0.00000°   x ≈ -0.00004 m   F ≈ 0.000 N
最大角度: 5.000°（初期値のまま、オーバーシュートほぼなし）
最大力: 43.43 N
```
