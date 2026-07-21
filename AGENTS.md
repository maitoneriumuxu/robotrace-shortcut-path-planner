# AGENTS.md

このリポジトリは、保存済み大会コースに対するショートカット経路生成、固定ATTACK速度モデルによる予測時間評価、比較画像確認だけに使う。

## 守ること

- `data/courses/normalized/`の31大会コースを削除・改変しない。
- `data/courses/README.md`と`data/courses/LICENSE.robotrace_course_cad`を残す。
- 参照先の`RoboTrace_LN5.xx`／`LN5.xx_GitHub`／`robotrace_course_cad`は読み取り専用とし、変更しない。
- 入力大会を2025年全日本へ固定せず、CLIの`--course`で任意の正規化TSVを扱う。
- 2025年固有の点index、画像座標、手作業で選んだ入口・出口をアルゴリズムへ埋め込まない。
- 局所ヘアピングループ専用ロジックは比較基準とフォールバックに残すが、大域探索の最終方式にはしない。
- 大域探索はPC品質確認用`reference`とRX651移植候補`embedded-lite`を分ける。
- referenceでもSciPy汎用非線形最適化、ニューラルネットワーク、乱数依存は使わない。
- embedded-liteは最大6100点、float32、固定配列化可能、再帰なし、大規模密行列なし、heap不要へ置換可能、決定論的な構成を保つ。
- 大域DAGの辺は必ず`start_index < end_index`とし、`source_progress_index`を単調増加にする。
- 大域経路は原ライン同一index法線へ限定しない。選択後は約10mm間隔へ再サンプリングする。
- 現在のFrenet法線方式は局所仕上げと既存最良フォールバックに残す。
- 速度モデルは`PlannerConfig`の固定ATTACK値を正とし、予測時間を良くする目的で変更しない。
- 1500deg/sは速度ハード制約ではなく、omega依存許容縦加速度が20m/s²となる閾値として扱う。
- AALP互換、実点間距離の前後速度スキャン4反復、始終端3.6m/sを維持する。
- 走行可能領域と白線重なりを分けて評価する。板境界がない大会は「板境界未確認」と明記し、実車外形未登録なら競技適合を断定しない。
- Elastic Band、旧#7、現在の長窓最良経路をフォールバックへ残し、新方式が遅い・無効なら現在最良へ戻す。
- 全31大会コースで異常終了、決定性、非悪化、フォールバックを回帰確認する。
- CSV、外部ログ読込、GUI、実機制御、ファームウェア組み込みは追加しない。
- Python予測結果を実走可能、実機保証、競技上安全、競技規定適合と断定しない。

## 比較画像

- 2025年全日本は`outputs/result.png`へ、原コース、現在経路、reference、embedded-lite、最終経路、アンカー、採用辺を表示する。
- 最大短縮区間に入口、出口、スキップ元区間、意図的白線交差を表示する。
- 速度、累積時間、現在経路との差、GFCP/AALP/加速/減速/最高速度の支配要因を表示する。
- 比較表へ予測時間、4秒差、現在差、長さ、幾何指標、交差数、探索規模、計算時間、判定を表示する。
- 全31大会のembedded-lite結果は`outputs/all_courses.png`へまとめる。
- 速度モデル固定値、板境界・実車外形の確認状態、実走保証でないことを画像内に明記する。

## 検証コマンド

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests -v
python -m robotrace_shortcut_lab --course data/courses/normalized/2025alljapan.tsv --mode reference
python -m robotrace_shortcut_lab --course data/courses/normalized/2025alljapan.tsv --mode embedded-lite
python -m robotrace_shortcut_lab --all-courses --mode embedded-lite
```
