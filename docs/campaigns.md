# USD シーンから教師データを作る

## 観察条件を設計する

対象物の見かけの大きさ、カメラ設置場所、照明、遮蔽、接触する部品、検出したい不良を記述する。実測した値、公開モデルの値、制作上の仮定に出典を付ける。現場写真を使う評価では RGB、カメラ内部行列、対象姿勢または対応マスク、深度の定義と単位を一組にする。別の作業環境で使う場合は、この組を交換する。[制作ガイド](authoring.md) · [実写との位置合わせ](registered-quality.md)

`campaign init` は元の USD を保持し、新しい設定を生成する。既存のリグや複数ステージがある場合は TOML を直接記述できる。既存の親座標系・姿勢・スケールを持つ対象に対して、ワールド座標の水平中心と最下点で配置する。全ての USD 層と参照用テクスチャはレンダーホストから読めるローカルファイルとする。コンテナ外のアセットには、それらの共通ディレクトリを `--asset-root` で指定する。

| 設定 | 作成者が決める内容 |
|---|---|
| `stage`, `camera_prim`, `width`, `height` | シーンと出力解像度。既存カメラまたはセッションに追加するカメラのパス |
| `classes`, `instances` | 1〜254 のクラス ID と、独立した prim ごとの ID。既存 prim も外部 USD の default prim も利用可能。背景は 0 |
| `cameras` | 視線、注視点の範囲、方位・仰角・距離、焦点距離の画素値、必要なら主点。配布例の最大俯角は 84 度 |
| `layouts` | 各インスタンスの支持高さと基準位置。個体ごとに水平ジッタ、ヨー角、出現確率を指定。干渉しない範囲・意図した遮蔽は制作者が設計する |
| `edits` | 光量や浮動小数プロパティ、USD の材質バインド、バリアント、表示状態、同時に切り替えるテクスチャ群。連続一様または列挙選択 |
| `poses` | 登録した実写やロボットの軌道に合わせたフレーム別のカメラ内部行列・ワールド変換と物体姿勢。サンプリングの代わりに使用 |

同じ seed とフレーム番号から乱数を生成する。カメラと配置の組を一巡しながら、数値を独立に抽出する。質量や重力による落下を止めた状態で画像と注釈を同期するため、動く物体の試験は別途 Isaac Lab で実行する。動的ロールアウトが必要な用途では、その軌跡を `poses` に供給する。[接触・把持の試験](contact-trials.md)

## 表面とセンサーを作る

色・粗さ・OpenGL 法線の登録された PBR 素材を用意する。下記は CC0 の紙素材を基に、傷の幅・長さ・深さ、透明デカールの配置と塗膜の厚さを実寸で指定する例。色の合成は線形空間で行い、傷とデカールの領域に合わせて粗さと法線も変える。各変形の採用値・出典ハッシュ・出力ハッシュは `finishes.json` に記録する。テクスチャのリピート長は対象メッシュの UV 実寸と合わせる。傷は表面の高さ・法線を変える。凹みや破れた角など外形に出る損傷は別メッシュで作り、USD バリアントで選ぶ。

```sh
uv run --locked python -m harness.finishes examples/campaign/paper.toml --out out/finishes
```

露出、トーンマップ、spp とバウンスを固定して RGB を撮影し、物体輪郭と距離を同じネイティブ Render Product から得る。別の光学系を再現するときは測ったレンズ・カメラ応答・周辺減光のプロファイルを適用する。RGB とラベルが同じ座標変換を通ること、視野外を有効画素から除くことを確認する。色ノイズやローリングシャッターまで要求する用途には、それぞれの測定が要る。[実測センサーの変換](sensor-model.md)

## 生成、再開、再撮影

作成済みの汎用設定ですぐに試す例。以下はコンテナにアクセスできる Linux RTX ホストで実行する。`HARNESS_KIT_CACHE` にホスト上の書き込み可能なディレクトリを指定すると、ジョブ間でシェーダーキャッシュを共有できる。

```sh
export ACCEPT_EULA=Y
uv run --locked python -m harness campaign capture examples/campaign/production.toml \
  --out out/multi --seed 20261125 --start 0 --count 128 --spp 16 --rt-subframes 3
uv run --locked python -m harness campaign capture examples/campaign/production.toml \
  --out out/repeat --seed 20261125 --start 0 --count 3 --spp 16 --rt-subframes 3
uv run --locked python -m harness campaign replay out/multi out/repeat \
  --count 3 --out out/comparison.json
```

`plan` は投入する件数を設計する。`run-plan` は同じ seed と設定 SHA を検証して範囲を実行し、実際に保存したフレームのハッシュを検査してから進む。同じコマンドを再実行すると完成したフレームを照合し、不足分を撮影する。複数ホストでは POSIX ロックと原子的な rename をサポートする共有ボリュームを用意し、同じ計画と出力に `--worker-count N --worker-index I` を与える。1 台の GPU では 1 ワーカーを使用する。ワーカー単位の試行ログ・終了コード・確認できた件数は `jobs/` に残る。全体の完成は `check --count 計画枚数` で確認する。

## データセットの受入

1. `campaign check --coco` が保存画像を開いてハッシュ、寸法、クラス・個体マスクの対応、可視矩形、距離を検証する。出力の `coco.json` は可視矩形と COCO の RLE 形式による個体マスク、`training.jsonl` は学習器用の矩形、`classes.png` と `instances.png` は画素単位の教師である。距離は光学中心からのユークリッド距離をメートルで格納する。
2. `campaign gallery` で RGB・色付きの輪郭・矩形・距離を原寸で確認する。出力の `review.json` にフレームごとの見え方と `pass` または `fail` を記入する。既存のレビューを生成時に上書きしない。
3. `campaign paired` は撮影姿勢が分かる独立シーンの実写と比較し、シルエット、光軸方向の深度誤差、対象内の明るさと色の差を測る。対象の物体・材質・照明が実運用と対応する参照セットを選ぶ。[位置合わせと閾値](registered-quality.md)
4. 合成画像だけで学習し、学習・校正に使用していない実写の撮影系列で検出性能を測る。データセットの `campaign_key`、学習・実写インデックスの SHA と分割方法をモデルと一緒に保存する。各クラスのサンプル数と合成ホールドアウトの値も報告する。[検出の実写評価](../evidence/transfer-published/README.md)
5. `campaign release` にプロジェクトで選んだ受入値、実写比較、検出指標、レビュー、独立再撮影を渡す。指定した証拠の欠落・不合格は終了コード 1 と判定 JSON になる。配布例の目標値はこのデモ用に選んだ値。[受入条件の例](../examples/campaign/release.toml)

```sh
uv run --locked python -m harness campaign release out/multi \
  --policy examples/campaign/release.toml --paired out/quality/quality.json \
  --transfer out/transfer/metrics.json --reviews out/review/review.json \
  --replay out/comparison.json --out out/qualification
```

`campaign survey` はレンダー前に、指定した全 index のカメラ×配置の件数と材質・照明・姿勢の分布を集計する。`rendered_frames` は 0 と記録する。センサープロファイルの動作確認用に、生成済みのフレームへ TUM の実測レンズを適用できる。プロファイルが再現するのは TUM のモノクロ撮影系である。

検出の一連の評価を動かす例。OpenCV と scikit-learn の版は `benchmarks/transfer/requirements.txt` で固定している。実写インデックスは画像パス、安定したフレーム ID、クラス・矩形を含む JSONL。結果にモデルと学習・評価入力の SHA を保存し、全ての学習画像について実写とのバイト重複を検査する。この SIFT ベースラインは模様を持つ剛体の試験に使える。実運用の検出器は、同じ COCO 出力と固定した実写分割で比較できる。

```sh
uv run --locked --group transfer python -m harness campaign transfer out/multi \
  --classes 2 3 5 --holdout-layout cross_pick --real-index /data/real/test.jsonl --out out/model
```

実測 RGB と明るさが合わないときは、光源を段階的に変えた RTX ブラケットを撮影する。`fit-exposure` は指定した校正写真の対象領域だけで光量の倍率を選び、`apply-exposure` は保持した別の写真の姿勢に倍率を適用する。校正と再評価が同じ写真を参照すると処理が止まる。[登録済み画像の評価と光量の固定](registered-quality.md)

物体の外形も変更する場合は、同じクラスの複数個体に `geometry_variant` を指定できる。選択した形状の寸法から支持高さを決め直し、ビューに写る表面と距離の整合を確認する。選択肢の USD とアトラスも撮影入力の SHA に含まれる。[異なる形状を同じクラスとして撮影する例](asset-families.md)

保存したデータから使用したソースへの対応は `campaign provenance DATA --config CAMPAIGN.toml --git-rev COMMIT --out receipt.json` で検査する。Git の指定があると、追跡対象の USD・テクスチャ・コードをそのコミットから読み出して撮影時の SHA-256 と比較する。外部ファイルはローカル内容を読む。`campaign check` は各フレームの注釈・各画素ファイルの SHA も `frames.sha256.jsonl` に出力する。

クラス間の平均では見落とせない品目がある場合は、受入 TOML の `[real_classes."5"]` などに `min_ap50` と `min_gt` を指定する。評価写真にそのクラスの正例が必要数含まれるかも判定する。配布した撮影例の性能基準は自社の安全・業務要件に合わせて定める。

別の登録シーンで外観を評価するときは、`[registered_inputs]` に評価姿勢リスト `poses` と閾値ファイル `quality_policy` を指定する。判定時に全評価写真の ID・撮影系列・実写 RGB の SHA、および閾値ファイルの SHA を照合する。

すでに `SemanticsLabelsAPI:class` を付けた USD では `campaign init assets/cell.usda --from-semantics --out jobs/cell` と書ける。同じラベルを継承する子は親の個体にまとめる。親子が異なる対象クラスを持ち、1 枚の個体マスクの所有権が重なるときは、撮影タスクごとに `--target` で対象を指定する。

紙や塗膜のベースは実寸でタイルできる素材を使い、個体固有の荷扱い印やシールを貼る面だけに専用の UV と寸法を与える。写真 HDRI で照らす場合は環境マップとドームの向き・ゲインを記録する。同じ視点、形状、材質で HDRI またはゲインを変えるとハイライトと影の変化を確認できる。[局所デカールの作成](surface-marks.md) · [写真 HDRI の比較](environment-light.md)

## GCP の RTX ホストで実行する

検証機はソウルの GCE `g2-standard-8`、NVIDIA L4 1 枚、Ubuntu 24.04 である。Linux ホストに NVIDIA コンテナランタイム、Docker、`uv` を用意し、取得した全アセットと出力を配置できるディスクを選ぶ。以下をリポジトリのルートで実行する。ホストの入力ディレクトリはコンテナでも同じ絶対パスに読み取り専用でマウントされる。

```sh
nvidia-smi
uv sync --locked
export ACCEPT_EULA=Y
export HARNESS_KIT_CACHE="$PWD/out/kit-cache"
uv run --locked python -m harness campaign run-plan jobs/parts/campaign.toml \
  --plan jobs/parts/plan.json --asset-root "$PWD" --out out/parts
```

Kit のキャッシュを同じドライバー・同じコンテナで再利用できる。今回の L4 では空のキャッシュでシェーダーのコンパイル時に Vulkan のメモリ確保が失敗した。同じ版のキャッシュを専用ディレクトリに復元して全シャードを完走した。別ホストへ持ち出す際は画像・注釈と各ジョブログをコピーし、`campaign check`、`provenance` を照合する。[検証機の実行条件と結果](../evidence/campaign-l4/README.md)

`campaign sensor` は測定済みのカラー光学系を全フレームに適用し、視野外を除いた矩形と個体マスクを生成する。凍結した実写テストを用いる学習の選択肢として `campaign transfer --detector ssdlite` がある。学習済みの画像分類器・物体検出器の重みを使わず、合成 RGB から検出器を学習する。[センサーを適用した学習セット](sensor-dataset.md) · [TorchVision 検出器の実行条件](neural-detector.md)

受入ポリシーの `[min_distinct_texture_sets]` に `paper = 8` などと指定すると、実際に選択した各チャネルの SHA-256 の組を数える。同じ画像に複数の仕上げ名を付けても 1 組になる。固定したビューのラベル検査には `[[appearance_pairs]]` を使う。[実物大の印刷を切り替える対照試験](surface-marks.md) · [テクスチャ 8 組の設定例](../examples/campaign/local-mark/coverage-policy.toml)
