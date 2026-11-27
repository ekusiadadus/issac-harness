# 写真 HDR による環境照明の比較例

`examples/campaign/warehouse-light/campaign.toml` は、独立した同一クラスの箱２個を、撮影された屋内 HDR 環境で照らす最小例。既存の `families/scene.usda` をサブレイヤーにし、形状、紙テクスチャ、クラス ID と２個体の prim を保持する。メートル、Z-up、固定カメラ 640×480、固定物体位置を使う。箱 A は cracker、箱 B は sugar を選ぶ。それぞれの形状バリアントは別々に変更できる。

HDR は [Poly Haven の Small Workshop](https://polyhaven.com/a/small_workshop) と [University Workshop](https://polyhaven.com/a/university_workshop) の写真 equirectangular 1K Radiance HDR。どちらも [CC0](https://polyhaven.com/license)。取得日、作者、API の元 URL、サイズ、元 MD5 と配布バイト列の SHA-256 は [manifest](../examples/campaign/warehouse-light/assets/manifest.json) に記録した。テクスチャは色ではなく線形 HDR 放射輝度データとして DomeLight の `inputs:texture:file` に渡し、`latlong` と Z 軸 0° 回転を固定する。撮影地点の世界方位は不明なので、0° は現場の方位合わせを意味しない。

`poses.json` の index 0–3 は同一カメラ、同一物体姿勢、同一材質と形状で、HDR ２種 × ドーム強度の倍率 0.5/1.0 を組み合わせた固定比較。USD の基準 `inputs:intensity = 2` に対し、実効値は 1 と 2 で１段の相対差になる。HDR 自体の絶対放射輝度と撮影現場の照度は校正していない。異なる HDR 間の明るさの比率も物理的な lux 比ではない。実現場の照度、撮影露出、表面反射率と HDR の色を測ってから、現場再現として光量を決めること。ここでの比較は合成レンダリング条件の差の検証に限る。

```sh
uv run --locked python -m harness campaign plan examples/campaign/warehouse-light/campaign.toml --frames 4 --shard-size 4 --seed 20261125 --out out/warehouse-light-plan.json
# Isaac Sim 実行環境で:
python runtime/capture_campaign.py examples/campaign/warehouse-light/campaign.toml --out out/warehouse-light --seed 20261125 --start 0 --count 4 --spp 16 --rt-subframes 3
uv run --locked python -m harness campaign check out/warehouse-light --start 0 --count 4 --out out/warehouse-light-check
```

撮影後は index 0 と 1 で同じ HDR の１段差、0 と 2 で同じ強度倍率の環境差を原寸で比較する。RGB、インスタンスマスク、矩形、撮影入力 SHA は `campaign check` と画像レビューで確かめる。同じ条件を別出力先に再撮影すると `campaign replay` で比較できる。レンダー前のローカル構成検査だけでは RTX の見えや露出は保証できない。

## 製品の撮影用ライトを合成する

`inspection.usda` は、写真 HDR のレイヤーに `campaign init` が生成した面光源と、別アセットのスキャン済み合板材質を重ねる。木目方向は台の長手、UV リピートは素材の 0.50 m に合わせた。カラー、粗さ、OpenGL 法線を使い、両軸の反復を有効にする。対象、ラベル、カメラ姿勢、HDRI 強度を固定し、手前の 0.363 × 0.254 m の面光源だけを 0 / 475 に切り替える。2 つの写真 HDR それぞれについて、キーライトのない側を対照にする。

```sh
uv run --locked python -m harness campaign capture examples/campaign/warehouse-light/inspection.toml \
  --asset-root "$PWD" --out out/workshop-inspection --seed 20261125 --count 4 --spp 64 --rt-subframes 4
uv run --locked python -m harness campaign check out/workshop-inspection \
  --count 4 --coco --out out/workshop-labels
```

この例は 1440 × 1080 の固定画角を使う。640 × 480 の元のカメラから内部行列の焦点距離・主点を 2.25 倍にし、画角を保った。[USD の合成レイヤー](../examples/campaign/warehouse-light/inspection.usda) · [4 ケースの撮影条件](../examples/campaign/warehouse-light/inspection-poses.json)
