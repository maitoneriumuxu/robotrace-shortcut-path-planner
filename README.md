# LN5.x ショートカット経路ラボ

改善要望を経路アルゴリズムへ反映し、幾何条件を評価して画像を1枚出すための最小構成です。

```powershell
$env:PYTHONPATH="src"
python -m robotrace_shortcut_lab
```

出力はスマホでも開ける`outputs/result.png`だけです。TXTログ、車速・ラップタイム、追従モデル、期待モデル、校正、ランキングは扱いません。

`embedded/shortcut_curvature_limiter.c/h`はRX651移植用の固定メモリ後処理です。全6100点の行列やheapを使わず、最大`dκ/ds`付近だけを161点窓で処理します。

## 現在の評価軸

- 最大オフセット: 100 mm
- 最小半径: 60 mm
- 曲率変化率: `dκ/ds [1/m²]`

一定速度で走るとき、角加速度は次式です。

```text
角加速度 α [rad/s²] = 速度 v² [m²/s²] × 曲率変化率 dκ/ds [1/m²]
```

したがって速度モデルを持たなくても、`|dκ/ds|`のピークを小さくすれば、同じ速度で必要な旋回トルクを減らせます。

現在の基準案は、長い直線を抽出してショートカットし、その前後を5次Hermite曲線で接続します。位置・接線・曲率を連続にすることで、直線と曲線の境界に生じる角加速度ピークを抑えます。

## 改善ループ

1. 改善要望を決める
2. `src/robotrace_shortcut_lab/algorithm.py`または`Settings`を変更する
3. 上の1コマンドを実行する
4. `outputs/result.png`を比較する

コースマップは`data/courses/normalized/`へ保存しています。
