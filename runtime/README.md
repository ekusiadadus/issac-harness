# RTX 撮影と Isaac Lab の読み込み検査

対象は Isaac Sim 6.2.0、Isaac Lab v3.0.0、Python 3.12。RTX 対応ホストにインストール済みの環境で実行する。

## Isaac Sim で撮影する

ハーネスのルートで次を実行する。各ケースとカメラを `scene.toml` から読み、RTX Path Tracing で描画する。フィルムバックに合わせた解像度、画像比較、対象領域の読取検査まで実行する。

```sh
uv run --locked python -m harness render examples/inspection --backend rtx \
  --isaac-python /path/to/isaac-sim/python.sh --capture-only --samples 64 --out out/inspection-rtx
```

Replicator を直接使う場合の例。Lib/Kit の初期化後に USD を開き、各カメラをウォームアップする。設定の読み戻し、RGBA の内容、保存 PNG の寸法とファイルハッシュを確認する。出力先は新しいディレクトリにする。

```sh
/path/to/isaac-sim/python.sh runtime/capture_isaac.py \
  --scene examples/inspection/scene.usda --output-dir /tmp/inspection-capture \
  --camera overview=/World/Cameras/Overview --camera detail=/World/Cameras/Detail \
  --renderer PathTracing --width 1536 --height 1024 \
  --spp 64 --warmup-frames 32 --rt-subframes 8 --max-bounces 4
```

`manifest.json` にアプリのバージョン、設定値、各カメラの PNG を保存する。記録する SPP は設定値と蓄積上限。RTX Real-Time と比較する場合は、直接実行の別の出力先を用意し、`--renderer RealTimePathTracing` を指定する。[Isaac Sim 6.2 の撮影 API](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/replicator_tutorials/tutorial_replicator_getting_started.html) · [SimulationApp の設定](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/py/source/extensions/isaacsim.simulation_app/docs/index.html)

## Linux / GCP の Docker で同じ版を実行する

ワークスペースのルートで実行する。`scripts/isaac_python.sh` に検証用 Sim と Lab の NGC イメージダイジェストを固定してある。NVIDIA の利用条件に同意した環境で `ACCEPT_EULA=Y` を設定する。

```bash
mkdir -p "$PWD/out"
export HARNESS_OUTPUT_ROOT="$PWD/out" ACCEPT_EULA=Y
uv run --locked python -m harness render examples/inspection --backend rtx --capture-only \
  --isaac-python "$PWD/scripts/isaac_python.sh" --width 1536 --samples 16 --out "$PWD/out/inspection-rtx"

scripts/isaac_python.sh "$PWD/runtime/generate_dataset.py" \
  --scene "$PWD/examples/inspection/scene.usda" --output-dir "$PWD/out/pilot" \
  --frames 4 --seed 20261125 --width 768 --height 512 --spp 16 \
  --warmup-frames 8 --rt-subframes 8

HARNESS_RUNTIME=lab scripts/isaac_python.sh "$PWD/runtime/lab_smoke.py" \
  --asset "$PWD/examples/inspection/environment.usda" \
  --marker-prim /World/Props/Carton/Base --drop-prim /World/Props/Carton \
  --steps 240 --drop-height 0.06 --settle-tolerance 0.006 --visualizer none \
  --report "$PWD/out/lab-carton.json"
```

各 `frame_NNNN` は同じ停止時刻の RGB、クラス ID の `uint32` マスク、ピクセル単位の tight bbox、カメラ原点からの距離をメートルで格納した `float32` 配列を持つ。`annotation.json` にクラス ID の対応、姿勢、材質・照明の抽選値と適用値、ファイルの SHA-256 が入る。同じ引数と seed で別の出力先に撮影して `dataset-check --replay` に渡す。端末の落下試験には `--marker-prim /World/Props/Handset/Body --drop-prim /World/Props/Handset` を指定する。

ネイティブの矩形 `[x_min, y_min, x_max, y_max]` は両端の画素を含む。学習用の固定 ID と Python の半開区間に変換する例を示す。元の RGB と距離配列は相対パスで参照するため、生成データと学習用フォルダを一緒に移動する。

```sh
uv run --locked python -m harness dataset-check out/pilot --replay out/pilot-replay \
  --report out/pilot-check.json --gallery out/pilot-gallery --training out/training
```

たとえば学習用の 1 フレームを次のように読み込める。RTX の元画像はアルファ付き PNG なので、3 チャンネル入力のモデルには RGB へ変換して渡す。

```python
import json
from pathlib import Path

import numpy as np
from PIL import Image

root = Path("out/training")
index = json.loads((root / "training.json").read_text())
frame = json.loads((root / index["frames"][0]["labels"]).read_text())
rgb = np.asarray(Image.open(root / frame["rgb"]).convert("RGB"))
mask = np.asarray(Image.open(root / frame["mask"]))
range_m = np.load(root / frame["distance_to_camera_m"], allow_pickle=False)
boxes = frame["boxes_xyxy_px"]
```
