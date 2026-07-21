# RX651向け曲率変化リミッタ

`shortcut_curvature_limiter.c/h`は、経路生成後の最大`dκ/ds`区間だけを処理する移植候補です。

- 最大6100点
- heapなし
- 追加作業RAM `SCL_WORK_BYTES = 5160 byte`
- 全点行列なし。161点の7重対角Cholesky解法
- 100mmオフセット、60mm半径を候補採用前に再確認
- 前倒し形状の涙滴補正は最初の2反復だけに限定
- 初期`dκ/ds`に応じて反復上限を5/8/12回へ切り替える
- 実走中や1ms割り込みではなく、モード5の一括生成時に呼ぶ

`scl_path_access`へ既存`VT_shortcut_path_buf`の読み書き関数を渡します。PC側で全コース評価が終わるまでは、ファームの`VT_SHORTCUT_ALGORITHM_SELECT`へ接続しません。

## 現在の確認結果

- CC-RX V3.07.00、RXv2/FPU指定で警告・エラーなし
- オブジェクトサイズ約8.2KiB、作業RAM5160byte以下
- CC-RX生成コードはソフトウェア`sqrtf`呼出しではなくRXv2の`FSQRT`命令を使用
- Cコードを直接実行した過去31コースすべてで、オフセット100mm、半径60mm、短縮率0%超を満たす
- 31コースすべてでC処理後の最大`dκ/ds`は処理前以下
- 2025全日本のC結果は短縮率6.383%、最大オフセット97.9mm、最小半径97.6mm、最大`dκ/ds` 173.8→134.6[1/m²]
- 最悪コースのホスト計測は、読み出し1,186,590回、書き込み7,159回、`sqrtf` 410,652回
- CC-RX出力では9箇所の平方根処理がすべてRXv2のハードウェア`FSQRT`命令になっている
- LN5ファームのテンポラリ複製へ最新C本体と`VT_shortcut_path_adapter.inc`相当を接続し、CS+ DefaultBuild成功（エラー0）
- 構造体計算上の追加静的RAMは5172byte（作業5152byte＋結果20byte）

再確認コマンド:

```powershell
python tools/check_ccrx.py
python tools/verify_embedded.py --compiler <tcc.exe> --image outputs/result.png
```

## ファーム接続時の順序

1. `shortcut_generate_elastic_band()`の半径走査を毎反復から10反復ごとへ間引く
2. `VT_shortcut_path_adapter.inc`相当を`VT_shortcut_path.c`へ置く
3. `shortcut_finalize_path()`の直前で`shortcut_apply_slew_limit(max_index)`を呼ぶ
4. 完了後に既存のyaw、radius、速度計画を再計算する
5. モード5実行後、`shortcut_slew_exec_us`が7,000,000以下かデバッガまたはDUMPで確認する

7秒はPC時間から推定せず、上記の実機計測を合格条件とする。現時点ではファーム本体は読み取り専用のため未接続。
