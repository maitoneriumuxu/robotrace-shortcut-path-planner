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
- 大域DAGの辺は必ず`start_index < end_index`とする。競技経路は始端から終端まで全LINE segmentを1区間も欠かさず進行順に通ることを必須とする。最終経路の`source_progress_index`は線形補間や`maximum.accumulate`で作らず、各姿勢で実際に接触した白線segment集合から単調DPで求める。通常の進行差は0または1とし、同一姿勢で中間segment全てへ実接触している場合だけ点列密度に応じた複数前進を許す。
- 大域経路は原ライン同一index法線へ限定しない。選択後は約10mm間隔へ再サンプリングする。
- 現在のFrenet法線方式は局所仕上げと既存最良フォールバックに残す。
- 速度モデルは`PlannerConfig`の固定ATTACK値を正とし、予測時間を良くする目的で変更しない。
- 1500deg/sは速度ハード制約ではなく、omega依存許容縦加速度が20m/s²となる閾値として扱う。
- AALP互換、実点間距離の前後速度スキャン4反復、始終端3.6m/sを維持する。
- 規定最大直径250mmの仮想車体と134.5mmラインチューブは「規定最大車体による非実車・非合法性確認の幾何下限」専用とし、reference、embedded-lite、最終採用の合法判定には使わない。
- 全車体外形`full_footprint_components_mm`と、白線接触を証明する物理部品`contact_witness_components_mm`を混同しない。最寄り白線距離だけでは合法判定しない。
- `design_confirmed=true`の接触証明部品で全姿勢の白線接触を証明した経路は「設計上合法」として比較できる。`as_built_confirmed=false`なら「実車確認済み合法」と表示しない。
- 白線幅19mmの領域と、2mm以下かつyaw変化1deg以下へ細分した全姿勢の接触証明部品を交差判定する。白線から完全離脱するcandidateは即不合格とする。
- 未来ラインへの同時接触は接触余裕の情報として記録してよいが、中間LINE segmentの省略理由にはしない。接触していないsegmentへの割当、segment飛越し、未通過LINEが1区間でもあるcandidateは即不合格とする。
- 走行可能領域と白線重なりを分けて評価する。全車体外形未確認時の板外判定は規定最大半径125mm円を使い、この円を白線接触判定へ流用しない。板内だけでは合格にしない。
- 接触証明部品が不明・`design_confirmed=false`なら新しい大域経路を採用せず、現在のFrenet長窓4.471秒経路へフォールバックする。
- Elastic Band、旧#7、現在の長窓最良経路をフォールバックへ残し、新方式が遅い・無効なら現在最良へ戻す。
- 全31大会コースで異常終了、決定性、非悪化、フォールバックを回帰確認する。
- CSV、外部ログ読込、GUI、実機制御、ファームウェア組み込みは追加しない。
- Python予測結果を実走可能、実機保証、競技上安全、競技規定適合と断定しない。

## 比較画像

- 2025年全日本は`outputs/result.png`へ、原コース、現在経路、競技無効な理論下限、設計上合法reference、embedded-lite、最終候補を表示する。
- 最小接触余裕、乗り換え、同時接触、最大差の位置を拡大し、接触証明部品と実際に接触した白線segmentを描く。
- 接触状態図へ実接触segment、DPで選んだsource progress、重なり面積、接触余裕、同時接触ライン数を表示する。未評価値を0として描かない。
- 比較表へ予測時間、合法性、robust警告、完全離脱回数、未通過LINE segment数、最小重なり・余裕、交差リスク、板境界、外形ソース、計算時間を表示する。
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
