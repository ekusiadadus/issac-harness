# GCP の L4 で生成したデータと実写評価

2026 年 11 月 25 日、ソウルの GCE で Isaac Sim 6.2.0、Replicator 1.13.38、Isaac Lab 3.0.0 を実行した。GPU は NVIDIA L4 1 枚。対象クラス、シーン、材質、センサー、実写の評価系列を設定から差し替えるハーネスの検証である。[実行環境](host/runtime.txt) · [自分の USD で開始する](../../README.md) · [撮影・判定の設定](../../docs/campaigns.md)

## RTX 教師データ

| 実際に生成・検査したもの | 観測結果 |
|---|---|
| 主データ | 960 × 720、異なる RGB 256 枚、可視個体マスクと矩形 1,469 件、クラス 6 種 |
| 観察条件 | 4 カメラ系統 × 4 配置で各組 16 枚。カメラ姿勢・焦点距離・対象の位置と回転・出現・照明・天板・壁・紙仕上げを変更 |
| 教師ラベル | RGB と同時取得した Replicator のクラスマスク・個体マスク・tight box・光学中心からの距離。保存後に可視矩形、クラス／個体 ID、距離、SHA-256 を全 256 枚で照合。COCO の可視個体マスクは RLE |
| レンダー | RTX PathTracing、16 samples/pixel、3 サブフレーム、4 バウンス。色処理と露出は固定。128 枚ずつ 2 シャードで完走 |
| 5 万枚の計画 | 391 シャードを設定。全 50,000 個の seed/index を CPU で集計し、16 組は各 3,125 件。実撮影・訓練に使った主データは 256 枚 |

各クラスの可視件数は箱 230、トレー 256、端末 241、クラッカー箱 250、砂糖箱 248、マスタード容器 244。初期の実装では画面に復帰した白い箱でラベルが欠落した。修正後、2 つの独立したネイティブ注釈と投影面の距離を照合し、同じフレームで 6 個体を復元した。[RGB・輪郭・距離のレビュー](review/index.html) · [全件の検査](native-labels/audit.json) · [検出マスクの修正前後](diagnostic-visibility) · [50,000 件の配置分布](plans/coverage-50000.json)

完全な撮影データはこの作業ディレクトリの [out/rtx-20261125/native](../../out/rtx-20261125/native) に保存した。Git には代表 8 フレームの 4 面と注釈、全 256 件の画像・注釈の SHA-256、COCO、取得時の 32 枚単位の転送記録がある。撮影プログラムとアセットの版は `cee79ff`。撮影が参照した 101 入力をこのコミットから照合した。[代表フレーム](native-examples/sample.json) · [全件の SHA](native-labels/frames.sha256.jsonl) · [転送記録](native-transfer-receipts.json) · [Git との照合](host/native-provenance.json)

## 実写で検出性能を測る

主データの 3 配置・192 枚を学習候補とし、残りの `cross_pick` 64 枚と実写 88 枚（BOP YCB-Video の別の 8 系列）で測定した。実写の正例はクラッカー箱が 22、砂糖箱が 44、マスタード容器が 11。実写の学習混入は RGB の全バイトハッシュで検査した。観察できるクラスの偏りがあるため対象 3 クラスの macro AP を報告する。[撮影系列の分割](../../examples/campaign/ycb/physical-split.json) · [SIFT の学習記録](native-model/model_manifest.json) · [評価入力の対応](native-model/real/lineage.json)

| 凍結した検出器・評価データ | macro AP50 | macro AP50–95 |
|---|---:|---:|
| SIFT、合成ホールドアウト 64 枚 | 32.44% | 14.95% |
| SIFT、実写 88 枚 | 24.94% | 11.77% |
| SSDLite、合成ホールドアウト 64 枚 | 19.84% | 9.40% |
| SSDLite、実写 88 枚 | 0.00% | 0.00% |

実写の SIFT AP50 はクラッカー箱 49.06%、砂糖箱 25.74%、マスタード容器 0.00%。被写体の印刷が部分的に見えても正しい大きさの矩形を引けない例がある。緑が正解、赤が予測。[実写評価の全クラス](native-model/real/metrics.json) · [実写の例](native-model/overlays/scene_54_frame_000001.jpg)

SSDLite320-MobileNetV3 はバックボーンも含めてランダム初期化した。L4 で 24 エポック、1 バッチ 8 枚、AdamW、学習率 0.001、seed 42。合成 192 枚を 190 秒で学習し、重みを凍結して同じテストを実行した。実写ではクラッカー箱に 2,756 件の予測が出たが、IoU 0.50 で正解に一致した検出は 3 クラスとも 0 件だった。評価では予測スコア下限を 0.05 に固定した。[学習条件と重みの SHA](native-neural/model_manifest.json) · [実写成績](native-neural/real/metrics.json) · [同じ実写でのニューラル検出器の予測](native-neural/overlays/scene_54_frame_000001.jpg)

## 同じ被写体・同じカメラの RGB-D 比較

YCB のスキャン形状・公開されたカメラ行列・物体姿勢を使い、RGB-D 実写と同じ被写体を同じ位置に置いて撮影した。最初の 6 枚は 2 系列のシルエット IoU が 0.988 と 0.945、光軸深度の絶対誤差中央値が 1.54 mm と 1.61 mm。一方、対象内部の sRGB 平均絶対差は 0.475 と 0.391 で白飛びが目立った。5 段階の光量を 2 枚の校正写真で比較して 1/16 に固定し、同じ 2 系列の重複しない 6 枚に適用した。画素誤差は 0–1 で、対象の可視領域の共通部分を 2 px 縮めて計算する。[初期評価](registered-quality/quality.json) · [倍率決定の全候補](exposure-freeze.json) · [校正写真と光量補正前後の RTX](exposure-comparison.jpg)

凍結した光量で撮り直した未使用の 6 視点では次の結果になった。左の系列では缶やドリルの局所反射、右の系列では洗剤ボトルの上部形状と明暗に実写との差が残る。背景は奥行きを合わせた支持面で代用し、外観スコアには物体の内部画素だけを使う。[未使用の全フレームと閾値](heldout-quality/quality.json) · [実写・RTX・輪郭の並列表示](heldout-quality/strips/00000003.png)

| 系列 | シルエット IoU | 光軸深度の絶対誤差中央値 | sRGB 平均絶対差 |
|---|---:|---:|---:|
| 50 | 0.990 | 1.55 mm | 0.178 |
| 51 | 0.940 | 1.61 mm | 0.244 |
| 事前の受入値 | ≥ 0.80 | ≤ 15 mm | ≤ 0.10 |

## センサーデータ

TUM Freiburg 1 の実測カラー内部行列と Brown–Conrady の歪み 5 係数を使い、主データの 256 枚を 640 × 480 に再投影した。RGB、クラス、個体、距離、有効領域のマスクを同じ光線で対応させ、可視矩形を再計算した。実写チェッカーボード 4 枚では、各画像の半分の角を使って姿勢を推定し、残り計 96 点の再投影誤差は中央値 0.23 px、95 パーセンタイル 0.60 px。元のカメラ校正に使われた写真上の整合確認である。対応するカラー実機のノイズ測定がないため、この派生データでは人工ノイズを無効にした。[校正写真の測定](sensor-color/calibration-check.json) · [全件の教師検査](sensor-color/labels/audit.json) · [レンズ変換後の画像](sensor-color/gallery/index.html) · [任意のプロジェクトへの適用](../../docs/sensor-dataset.md)

## 再現性と受入判定

同じ seed・設定・入力 SHA で、別 Kit プロセスから最初の 3 枚を再撮影した。RGB の 0–1 sRGB RMSE は 0.00315、0.00584、0.00410。可視個体の最小 IoU は全て 1.0、共通対象内の距離差は全て 0 m。完了フレームを再開すると元画像の内容を照合してスキップする挙動も確認した。NVIDIA 公式の静的 Isaac Sim Validator と SDG 差分 GIF スクリプトを固定リビジョンで実行した。[再撮影結果](replay.json) · [RGB 差分を10倍した GIF](upstream/replay-rgb.gif) · [クラスマスクの差分 GIF](upstream/replay-semantic_segmentation.gif) · [公式 validator のログ](upstream/validator-campaign.log)

**この主データの受入判定は不合格。** フレーム数、クラスと材質の出現数、画像と学習データの来歴、独立再撮影は通過した。実写の sRGB 誤差、事前指定した実写 AP50 ≥ 0.50、目視 4 枚中 3 枚が不合格だった。反復した荷扱い印、金属の研磨目と反射の出方に問題がある。後続の別サンプルの改善はこの判定に含めない。SIFT がこの受入判定に使った検出器であり、SSDLite は同一分割を用いた追加測定。[受入判定と証拠の SHA](qualification.json) · [設定](../../examples/campaign/release.toml) · [画像ごとの判定理由](review/review.json)


## 別の USD、同一クラスの複数個体と照明

別の USD に同じクラスの製品を 2 個登録した。各個体を赤いクラッカー箱または白い砂糖箱に差し替え、4 通りの組み合わせを RTX で撮影した。形状を変えても各個体の可視マスク・矩形と安定 ID を追える。[独立アセットの作成方法](../../docs/asset-families.md) · [全 8 枚・16 個体の検査](family-studio-labels/audit.json)

当初の支持面は、UV が単位タイルを超えた部分が黒かった。実寸 0.16 m の紙タイルで両軸の反復を明示した後、同じ 8 視点で 2 灯の光量を 1/4 にして撮り直した。支持面のいずれかの RGB チャンネルが 250/255 以上になる画素の割合は、8 枚の中央値で 95.16% から 0.00% に下がった。支持面の sRGB 中央値は 1.00 から 0.622。両撮影で個体マスクの差は全 8 枚で 0 画素だった。[USD テクスチャの検査](material-qa/paper.json) · [8 組の測定値](material-qa/studio-exposure.json) · [補正前](family-wrap-gallery/index.html) · [補正後](studio-gallery/index.html)

同じ 2 個体を、撮影された工房 HDRI 2 種と 0.50 m 周期の合板スキャン材質で 1440 × 1080・64 spp にした。固定視点で面光源を 0 / 475 に切り替えた全 4 枚では、両環境とも印刷と木目の明暗が変わり、各 2 フレームのネイティブ個体・クラスマスクと距離は完全に一致した。HDRI や面光源の絶対照度はこの例で校正していない。[照明と材質の合成例](../../docs/environment-light.md) · [各画像の原寸比較](workshop-gallery/index.html) · [照明比較の計測](material-qa/workshop-2.json)

## 接触、滑り、把持

同じ GCE インスタンスの Isaac Lab で各試験を別プロセスで各 360 ステップ、1/120 秒刻みで実行した。横衝突と滑走は GPU 物理、運動学的な 2 本の指を動かす試験は CPU 物理。Lab の配布版は 3.0.0、ロードした拡張の版は 28.0.0。[4 試験の全軌跡とログ](physics/final) · [別のコライダーと治具を指定する方法](../../docs/contact-trials.md)

| 試験 | 実際の観測 |
|---|---|
| トレー縁への横衝突 | 入射 0.712 m/s、反射 −0.142 m/s、有効速度比 0.200 |
| 端末の滑走 | 11 点の減速区間から求めた有効動摩擦係数 0.655。速度回帰の RMS 誤差 0.0000045 m/s |
| 指を閉じて持ち上げる | 保持高さ平均 179.39 mm、開放後の落下 179.39 mm |
| 指を開いた対照 | 同じリフト軌道を動かしても箱は支持面にとどまった。上昇は 0.001 mm 未満 |

最初の把持試験では、reset 後の指が原点に集まり、開いた対照でも箱が飛んだ。指の位置・ターゲット・速度を計測前に確かめてから再試験した。ここで求めた摩擦と反発はこのコライダー・材質・solver に対するシミュレーション値で、現物の係数測定はこの試験の対象外。[初回の異常](physics/cpu-initial) · [原因切り分け時の生ログと軌跡](physics/diagnostic-stages.tgz)

制作と比較の判断は Amazon Robotics の形状・材質・デカール・損傷を組み合わせた制作例、NVIDIA の Replicator、SimReady、Isaac Sim の公式 skills、SceneSmith の画像批評・物理検査を参照した。機種あたり 5 万枚超の公表値の出典は Amazon Devices & Services の製造検査事例である。[出典と実装の対応](../../docs/references.md)

## 印刷や部品の着脱で教師ラベルが欠ける問題

物体・カメラ・照明を固定した近接撮影で、面上の印刷を非表示から表示に戻すと、元のランタイムはその下の **13,738 画素を箱から背景へ**変えた。クラスと個体のネイティブ出力が両方とも同じ誤りを返したため、2 面の相互比較だけでは見逃していた。逆順では再現せず、透過表現を不透明な切り抜きメッシュに換えただけでも 13,670 画素が欠けた。[修正前の対照](material-qa/alpha-mask-invariance.json) · [不透明メッシュだけを変えた対照](material-qa/opaque-only-invariance.json) · [撮影順と配置方式の切り分け](material-qa/registration-controls/index.json)

撮影時には対象配下の部品を常に RTX へ登録し、不要なフレームではカメラの背後へ退避させる。画像上で印刷を「なし→あり→なし」にした 3 枚で、両方向ともクラス・個体の差は **0 画素**。RGB の差が 0.10 を超えた箱の画素は順に 13,016・13,021、半径距離差の最大は 0.851 mm で、表面からの設定距離 0.8 mm と整合した。マークの投影領域 14,228 画素は全て箱の ID のままだった。[RGB と実マスクの接写](material-qa/decal-control.png) · [対照試験の受入結果](mark-qualification/qualification.json) · [入力・コード 104 件の照合](host/label-fixed-provenance.json)

現在の学習サンプルでは、スキャン由来の基材に seed 付きの傷を焼き込み、印刷は対象の 1 面に 1 回だけ付ける。8 組の色・粗さ・法線は画素データのハッシュで区別する。固定対照や材質数のポリシーは他の USD と対象クラスにも使用できる。[材質のレシピ](../../examples/campaign/local-mark/paper.toml) · [各実寸傷と出力ハッシュ](../../examples/campaign/local-mark/finishes/finishes.json) · [適用手順](../../docs/surface-marks.md)

この修正版でさらに 960 × 720 の 32 枚を GCP で撮影した。4 カメラ系統・4 配置、可視個体 186 件、内容の異なる紙仕上げ 8 組、印刷の有無 2 通りを保存後に確認した。4 枚の目視では、印刷の反復は解消し、俯瞰の金属面には強い反射と規則的な研磨縞が残った。この 32 枚による検出器の再学習・実写 AP 測定は実施していない。[全 32 枚の保存先](../../out/rtx-20261125/production-fixed) · [材質と撮影条件の網羅検査](production-coverage/qualification.json) · [目視の判定と画像](production-gallery/index.html) · [Git と入力 103 件の照合](host/production-fixed-provenance.json)

## 実行記録

主データの生成コードは `cee79ff`、部品の着脱・実ファイルによる材質数検査・修正版 32 枚の生成コードは `dc99789`。ローカルは 99 テスト成功、Torch を必要とする 1 件をスキップ。Torch の対象テストは GCP のコンテナで 2 件とも成功した。NVIDIA 公式スクリプトで最終撮影コードを検査し、静的 Level 2 のエラー・警告は 0 件だった。[テストログ](host/local-tests.log) · [GCP の Torch テスト](host/torch-targets.log) · [USD シーン検査](host/scene-check.json) · [Kit の生ログ 31 ファイルと取得 SHA](logs/receipt.json) · [生ログ一式](logs/kit-logs.tar.gz)
