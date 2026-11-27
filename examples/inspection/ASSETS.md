# 梱包作業台の素材と寸法

梱包箱と携帯端末の外観・ラベル検査を想定して制作した作業台。USD は Z-up、1 m/unit。天板は 1.65 × 0.85 m、箱は約 0.32 × 0.26 m。寸法・配置・照明はこのサンプルの設計値。

## スキャン素材

Poly Haven の 1K 配布画像を同梱した。各原本を API の MD5 と照合し、ベースカラー・粗さ・金属度は JPG、OpenGL 法線は PNG のまま配置した。以下の素材のライセンスは [Poly Haven CC0](https://polyhaven.com/license)。

| USD の材質 | 制作者と素材 | 実寸幅・用途 |
|---|---|---|
| Concrete | Rob Tuytel [Concrete Floor Painted](https://polyhaven.com/a/concrete_floor_painted) | 2 m。床に 2 m 周期で割当。作業物に視線が向くよう USD で床色のコントラストを圧縮 |
| Wood | Rob Tuytel [Plywood](https://polyhaven.com/a/plywood) | API 実寸幅 0.5 m。天板上面に 0.5 m 周期で割当。水平木目を天板の長辺 X 方向へ合わせた。床板の画像は使用しない |
| Steel | Rob Tuytel [Metal Plate 02](https://polyhaven.com/a/metal_plate_02) | 2 m。小さな金属試片に割当。酸化部と素地の違いを金属度マップで表現 |

箱の表面には [ambientCG Cardboard 002](https://ambientcg.com/a/Cardboard002)、右下の局所的な紙繊維の擦れには [Paper 005](https://ambientcg.com/a/Paper005) の 1K-JPG 配布物から Color、Roughness、NormalGL を無加工で同梱した。[ambientCG の CC0 ライセンス](https://docs.ambientcg.com/license/) に従う。取得した ZIP の SHA-256 は順に `c9e666f77d36889873c5e9cfcdd470f153802ac3025cb0d3f765b64725184f2c`、`5e9d07bf4f3718b97c107f6c54943a7779e19a2051ccead72e7740efb9d149c0`。ambientCG の実寸指定は両素材とも 0 のため、箱は 0.45 m、局所擦れは 0.06 m の制作上の UV リピート幅を使う。取得日は 2026-11-25。

## 制作した素材

- **樹脂・紙・壁塗装・テープ** は seed 固定の 512 × 512 PNG。箱の段ボールと局所擦れのみ上記 ambientCG 素材へ変更した。箱の 3 mm 側壁上端には 2 枚のライナーと連続した正弦波形の芯紙を幾何として作った。[FEFCO 2015 Table 1](https://www.fefco.org/download/file/fid/2135) の B フルート（150 山/m、ライナーを除く高さ 2.4 mm）を採り、周期 6.67 mm、片振幅 1.2 mm、約 8 分割/周期とした。正面フラップの切断面も上下の薄いライナーと XZ 面の芯紙に分け、前端へ露出させた。前右端の透明テープは凹輪郭内に三角形分割した薄いメッシュで、色は下地と同じ Cardboard002、横しわの法線と粗さは seed 固定の PNG、PreviewSurface の不透明度は 0.38。擦れ跡はラベル右端より外側の細い紙繊維の小領域とした。ラベル台紙の UV 周期は 0.25 m。テープと摩耗の形・しわの配置は制作上の推定。
- **ヘアライン金属** は別の 512 × 512 素材。研磨方向に沿う粗さと微細法線、金属度 1 をトレー、定規、脚部に使った。脚の側面 UV は研磨目が鉛直 Z に沿うよう回転した。トレーのリピート幅は推定 1.5 m。
- **ラベル** は 130 × 59 mm、印刷面は 1,300 × 590 px。品番、ロット、EAN-13 の黒線を一枚の平面に印刷した。図柄は Pillow と ZXing-C++ で生成。白地の余白を設け、撮影 PNG 上でも番号を照合する。対象番号はサンプル用の `0426112500014`。

カラー画像は sRGB、粗さ・金属度・法線は raw。符号なし法線の RGB を 2 倍して 1 を引き、接線空間の範囲に戻す。全ての画像を `geometry.usdc` からの相対パスで参照する。

Plywood 原本の MD5（[Poly Haven files API](https://api.polyhaven.com/files/plywood) と照合）：`plywood_base.jpg` 8032b768029adffbbfb223dd890102de、`plywood_rough.jpg` d46c674551da8c15b2a82000e22fc60b、`plywood_normal.png` 6d3ed6894e61733de595a40f68dd1044。

## シーンの構成と再生成

`scene.usda` は環境、照明、カメラを合成する。Lab の入口はライトと PhysicsScene を含まない `environment.usda`。ライトは `lighting.usda`、重いメッシュ配列は `geometry.usdc` および独立した `assets/{carton,handset,tray}/*.usdc` に格納した。各アセットは単一の default prim に剛体、質量、意味ラベルを持ち、必要な材質とテクスチャを同じフォルダに含む。`geometry.usdc` はそれらを元の `/World/Props/` 配置へ参照する。トレーの四辺、箱の四辺にもコライダーを付けた。質量と摩擦は制作上の推定で、静止位置の設計値と実際の物理安定性は別途 Lab で検証する。

箱と携帯端末には `surface_finish` variant がある。箱は `standard` / `worn`、端末は `standard` / `matte` で、粗さ画像の入力変換を変える。アセット全体のコード・形状にはライセンスを表明していないため、メタデータの `asset_license` は SPDX の `NOASSERTION`。同梱 Poly Haven の画像は上記 CC0。

カメラは 36 × 24 mm のゲート、全景 42 mm、接写 58 mm。天井の面光源と別に、箱の前面を向いた 0.72 × 0.46 m のソフトボックスを置いた。ドーム光は Poly Haven の [Empty Warehouse 01](https://polyhaven.com/a/empty_warehouse_01)（Sergej Majboroda、CC0）の 1K HDRI。L4 の初回 RTX 画像では全チャンネル最大 77/255、フレーム中央値 19、印刷紙の白が約 70 と暗かったため、全光源の `intensity` に共通の 3.5 stop（約 11.31 倍）を掛けた。これはスタジオ画像の制作上の補正値であり、現場の実測照度ではない。減光ケースは上部光源のみさらに −2 EV。カメラ露出と相対的な光源配分は固定。接写の判定対象は `/World/Props/Carton/Label` で、実際の印刷面はその手前の `LabelPrint`。

ハーネスのルートで `uv run --locked python scripts/build_example.py` を実行すると、同梱のスキャン画像を使って配布用サンプルを再生成する。新しい案件では `harness init` で複製したディレクトリを編集する。
