# 撮り直しと修正の記録

| 観察 | 取得した証拠 | 変更 |
|---|---|---|
| 画面全体が暗く、白いラベルの階調を使い切れていなかった | [初回照明の RTX 全景](underexposed-overview.png)・[実行設定](underexposed-manifest.json)（`f764955`） | カメラの露出を固定し、各光源の強度に共通の 3.5 stop 増分を適用。天井光だけを変える比較は −2 EV を維持 |
| 段ボール右下の損傷が矩形に見え、正面からは手前のフラップの内部が見えなかった | [修正前の RTX 接写](rectangular-wear.png)・[実行設定](rectangular-wear-manifest.json)（`93b689b`） | CC0 の紙の素材、小さく不規則な擦れ、透過するテープ、2 枚のライナーとその間のフルートを追加 |
| 初期のフルートに開いた面と索引のない UV があった | [NVIDIA の指摘](open-flute/usd-validation/nominal.json)（`0128230`） | 底面を閉じて法線と索引付き UV を生成。修正後の検査は [最終 USD 検査](../scene-check.json) に記録 |
| テープを斜めに横切る三角形が残り、端の紙剥がれがシール状に見えた | [実寸の RTX 接写](overlapping-tape/nominal/detail.png)・[レビューの不合格記録](overlapping-tape/acceptance.json)（`442ff49`） | 輪郭の内側で三角分割し、フィルムを外壁に密着。テープには下地のスキャン色としわの粗さ・法線を与え、剥離を細くした |

初回の Replicator 読み出しは空配列を検出して失敗した。レンダープロダクトの全アノテータに同じ時刻のバッファが届くまで待機する処理を追加した。Lab の最初の読み出しでは 60 mm 持ち上げた USD の記述値が 240 ステップ後も変わらず、落下検査が失敗した。実際の PhysX テンソルから位置・速度を記録するように修正した。

同じ seed の独立プロセスで、端末の生セマンティック ID が 4 から 3 に変わった実例も保存した（`442ff49`、フレーム 0）。元の ID→クラス名で復号すると箱・端末の画素は一致する。学習用の出力はこの対応表を使って固定 ID に変換する。[初回の対応表](semantic-id-allocation/pilot/annotation.json) · [再実行の対応表](semantic-id-allocation/replay/annotation.json) · [4 枚の照合結果](semantic-id-allocation/dataset-check.json)
