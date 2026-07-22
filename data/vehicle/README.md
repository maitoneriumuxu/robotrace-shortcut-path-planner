# LN5外形と白線接触証明部品

`LN5_footprint.json`は、全車体投影と白線接触を証明する物理部品を分離する。

- 座標系は`+X=前方`、`+Y=左方`。
- `full_footprint_components_mm`: タイヤ、基板、モータ、吸引機構を含む全車体床面投影。現在は未確認で空。
- `contact_witness_components_mm`: 白線と重なることで車体全体の接触を証明できる、確実に存在する部分。
- `design_confirmed`: 製作可能な設計としてユーザー確認済み。
- `as_built_confirmed`: 製作後に外形と取付位置を実測済み。

現在のcontact witnessは、車体原点を横切る前後10mm・左右200mmの横バーで、頂点は`[-5,-100] [5,-100] [5,100] [-5,100]`。出典は`user-approved planned transverse contact bar, 2026-07-22`で、`design_confirmed=true`、`as_built_confirmed=false`である。

白線接触はこの横バーと幅19mm白線領域の交差で判定する。全車体外形が未確認なので板外判定には半径125mmの保守円を使う。この円を白線接触の証明には使わない。

現段階の合格経路は「設計上合法」であり「実車確認済み合法」ではない。製作後は全車体外形、横バー実測頂点、取付原点を記録して`as_built_confirmed=true`へ更新する。
