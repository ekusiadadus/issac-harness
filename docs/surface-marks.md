# Localized surface marks

Build the example from the repository root:

```sh
uv run --locked python examples/campaign/local-mark/build.py
uv run --locked python -m pytest -q tests/test_surface_mark.py
```

The generated `examples/campaign/local-mark/production.toml` uses the same scene geometry, six instance definitions, camera and light distributions, layout anchors, and paper choice IDs as `examples/campaign/production.toml`. Its stage is a small USD layer over the original inspection scene. One RGBA handling graphic occupies a 60 × 37 mm area, 0.8 mm above `/World/Props/Carton/WallFront` face 31 at face coordinates `(0.17, 0.40)`. These dimensions and position are **art direction estimates**, not measurements of a physical package. The source graphic is `examples/campaign/handling-mark.png`; the mark follows the carton transform and uses its enclosing instance as the expected semantic owner. The UV spans the graphic exactly once. The placement has been checked against `/World/Props/Carton/LabelPrint`; the earlier central location was obscured by that label.

The production example has eight distinct surface finishes. Its recipe generates eight registered base-color, roughness and normal map sets from the scans with seeded scuffs only. The localized print toggles independently in the scene. Each sampled finish and the actual print selection are retained for replay. The fixed close-up uses finish07 for all three frames.

`harness.surface_mark.add_mark` requires an explicit mesh path and quad face index, owner path, image path, dimensions, face position, and normal gap. It rejects degenerate, nonplanar and nonconvex faces, marks that cross the selected face boundary, invalid dimensions, and duplicate names. Face winding and USD orientation control the outward normal. For other assets, select an outward-facing planar quad. Image review in Kit/RTX remains necessary to judge visibility, blending and z-fighting; CPU composition only proves the authored structure and path resolution.

For a separate USD stage, create a new composing layer directly:

```sh
uv run --locked python -m harness.surface_mark source.usda marked.usda \
  --mesh /World/Props/Object/Surface --face 0 --owner /World/Props/Object \
  --texture mark.png --width-mm 60 --height-mm 37
```

The output layer references the source stage and texture by relative path. The mark is authored below the selected mesh, so nested transforms follow it. Dimensions and the normal gap are computed in world metres using stage units and the full transform; the selected quad must be finite, planar and convex. `leftHanded` orientation reverses the offset normal. The tool leaves the source stage untouched and refuses an existing output file.

The alpha channel is converted to opaque physical cutout geometry at a hard cutoff of 128 (values 128–255 are kept). Runs of kept pixels are merged vertically into quad faces; each face retains the source image's UV coordinates, including transparent holes. The preview material samples RGB but does not connect `opacity` or set `opacityThreshold`. Pass `--alpha-cutoff N` to the CLI or `alpha_cutoff=N` to `add_mark` to choose a different integer threshold from 1 to 255. Hard edges can look jagged at close range; check the final RTX image at the intended observation distance.

Repeat `--keep-clear /absolute/prim/path` to reject overlap with existing parts. It compares the oriented mark region with the named prim's world bound, allowing a 1 mm normal margin. This is conservative for rotated or irregular obstacles and does not replace an image review.

For the controlled close-up, capture indices 0 through 2 of `examples/campaign/local-mark/compare.toml`. The front camera, lights, finish and object poses stay fixed. The mark transitions from hidden to visible to hidden. The other five targets are outside this view. Both transitions are measured against the same appearance policy.

### 固定撮影で印刷と教師マスクを同時に確認する

同じ対象・物体姿勢・カメラ・露出を固定し、マークだけを切り替えた画像を撮る。下記では sRGB の平均チャンネル差が 0.10 を超える画素を数え、同じ箱に属する範囲内で最低 1,000 画素が変わることを要求する。厚さに伴う半径距離差は最大 2 mm とした。クラスと個体のラベルには画素単位で一致を求める。部品のサイズと観察距離に合わせて閾値を決める。

```sh
uv run --locked python -m harness campaign capture examples/campaign/local-mark/compare.toml \
  --asset-root "$PWD" --out out/print-control --seed 20261125 --count 3 --spp 32 --rt-subframes 4
uv run --locked python -m harness campaign release out/print-control \
  --policy examples/campaign/local-mark/appearance-policy.toml --out out/print-qualification
```

単独の比較は `campaign label-invariance` でも実行できる。通常の撮影計画に固定対照を含めた場合は、同じ `[[appearance_pairs]]` をプロジェクトの受入ポリシーに加える。
