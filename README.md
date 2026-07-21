# Robotrace Shortcut Path Planner

2025年全日本ロボトレースの保存済み点列だけを入力し、ショートカット経路と過去LN5.x ATTACK相当の予測時間を比較する検討環境です。

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests -v
python -m robotrace_shortcut_lab
```

結果は`outputs/result.png`へ生成します。CSV、外部ログ、GUI、実機制御、ファームウェア組み込みは扱いません。

## 比較する4経路

1. 原コース
2. 現行Elastic Band相当
3. 基準コミット`d5a5a55`の旧時間選択型#7
4. 長窓Frenet探索後の最良経路

長窓探索は原ライン、Elastic Bandを原ライン法線へ射影したオフセット、旧#7の3初期値を試します。高曲率・低速で曲率符号が交互に現れる密集区間を自動検出し、200/500/800 mmのraised cosineを固定候補として評価します。点は常に次の同一index法線上に置き、最近傍探索や別区間への再射影は行いません。

```text
q_i = p_i + d_i n_i
```

Elastic Band自身と旧#7も最終候補へ残すため、採用経路の予測時間は少なくともElastic Bandおよび旧#7より悪化しません。候補は1本ずつ生成・評価し、保持するのは最良候補の記述子だけです。

## ATTACK速度モデル

参照元は`RoboTrace_LN5.xx`相当の読み取り専用リポジトリにある次のコミットです。

- `ace1bcb0a16e30032d1317becee519023847e988`
- `14afdb2dcd6e7b27dd42a53c80f7adfd16691bd4`

全日本ATTACKパラメータ3を基準に、最低速度だけユーザー指定の3.6 m/sへ変更しています。

| 設定 | 値 |
|---|---:|
| `R10_speed` | 3.0 m/s |
| 最低速度、始端・終端速度 | 3.6 m/s |
| 最高速度 | 13.0 m/s |
| `speed_root` | 0.33 |
| 最小／最大加速度 | 20.0／55.0 m/s² |
| 減速度 | 55.0 m/s² |
| `break_kp` | 1.00 |
| `max_AALP` | 100（後述のファーム数値単位） |
| 加速度補間omega閾値 | 300／1500 deg/s |
| `jerk` | 3.0（Dutyモデルがないため予測時間には未使用） |

GFCPは実経路の半径から次で求め、3.6～13.0 m/sへクランプします。

```text
v_GFCP = 3.0 × abs(radius_mm / 100.0) ^ 0.33
```

1500 deg/sは速度ハード制約ではありません。`omega = v × curvature`をdeg/sへ変換し、許容縦加速度を300 deg/s以下で55 m/s²、1500 deg/s以上で20 m/s²、その間は線形補間します。実点間距離による前向き加速スキャンと55 m/s²の後向き減速スキャンを固定4回繰り返します。

## AALP互換の単位と再現

参照コミットの`marker_less.c::Angular_Acceleration_Limit_Planner()`は、`SEARCH_RAN_SPEED=3.6 m/s`と`v_ref × sqrt(alpha_limit / alpha_meas)`を使い、GFCPとAALPの小さい方を採用しています。

一方、`mpu_calculation.c`の`MST.yawAccel`は1 ms周期の`gyro_z[deg/s]`差を時間で割らずに保存し、`log.c`はそれを`yawAccel[deg/s2]`と出力しています。したがってログ見出しと実数値の次元表記は一致せず、`max_AALP=100`と比較される実装上の数値単位は「deg/sの1 ms当たり変化量」です。100をrad/s²やdeg/s²へ読み替えていません。

新経路には探索ログがないため、定速3.6 m/s探索（縦加速度0）として次を使います。

```text
alpha_rad_s2 = 3.6² × dcurvature/ds
alpha_firmware = abs(alpha_rad_s2) × 180 / π × 0.001
v_AALP = 3.6 × sqrt(100 / alpha_firmware)
v_base = min(v_GFCP, v_AALP, 13.0)
```

主結果はGFCP+AALP互換モデルです。GFCP単独結果も計算しており、今回のコースではAALP支配点が0点だったため両者の差は0.02 ms未満でした。

## 幾何制約

改善後経路では100 mm最小半径をハード制約にしません。半径低下はGFCP、AALP、omega依存加速度を通じて予測時間へ反映します。代わりに次を検査します。

- 横オフセット75 mm以下
- 最大点間距離20 mm以下
- 1点ごとのオフセット変化10 mm以下
- 非有限値なし
- 原コース対応線分に対する逆行なし
- 原コースより自己交差を増やさない
- 最大`|dκ/ds|` 180 1/m²以下（旧#7実測82.6の約2.2倍とした数値異常ガード）
- 始端・終端の元ライン復帰

旧#7を基準表示として再現する処理だけは、基準コミットの100 mm半径修復を`legacy_min_radius_mm`として保持します。改善後の長窓更新には適用しません。

## 現在の結果

| 経路 | ATTACK+AALP予測 [s] | GFCP単独 [s] | 長さ [m] | 最小半径 [mm] |
|---|---:|---:|---:|---:|
| 原コース | 6.132 | 6.132 | 35.024 | 102.3 |
| Elastic Band | 5.189 | 5.189 | 33.222 | 100.2 |
| 旧時間選択型#7 | 5.062 | 5.062 | 33.007 | 100.1 |
| 改善後 | 5.011 | 5.011 | 32.811 | 85.3 |

改善後はElastic Bandより0.178秒、旧#7より0.051秒短い予測です。これはPython上の比較結果であり、実走可能性、実機性能、追従余裕、競技上の安全を保証しません。詳細な中間比較、支配区間、計算量は`docs/handoff-2025alljapan.md`に記録しています。

コースデータの出典とライセンスは`data/courses/README.md`と`data/courses/LICENSE.robotrace_course_cad`を参照してください。
