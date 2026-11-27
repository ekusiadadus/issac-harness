# issac-harness

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**English** | [日本語 (README.ja.md)](README.ja.md)

OpenUSD and Omniverse Replicator fail silently. A stage with broken texture references, 100× lens/aperture unit typos, 8 mm resting gaps above a table, or mid-run segmentation holes still exits `0` and renders plausible-looking PNGs.

`issac-harness` turns those silent defects into fast, deterministic `exit 1` checks for **NVIDIA Isaac Sim 6.2.0** and **Isaac Lab v3.0.0**. It separates sub-minute offline checks on laptop CPU / CI (`usd-core`, isolated `usd-validation-nvidia`, SimReady Robotics-Prop, and Blender Cycles / Storm lookdev) from digest-pinned NGC Docker runs on Linux RTX hosts (Path Tracing dataset capture, cross-process replay verification, PhysX contact/settling trials, and locked real-image release gates).

![Fixed close-up decal invariance check: toggling visibility mid-run erases 13,738 carton pixels underneath the surface mark in both semantic and instance masks (center, 490/14228), while keeping child marks registered and parking them behind the camera preserves all 14,228 pixels (right)](evidence/campaign-l4/material-qa/decal-control.png)

Complete NVIDIA L4 benchmark outputs—including synchronized RGB/mask/depth captures, registered BOP YCB-Video RGB-D comparisons, PhysX contact traces, and an actual failed `campaign release` audit—are archived in [`evidence/campaign-l4/`](evidence/campaign-l4/README.md) and [`evidence/gcp-l4/`](evidence/gcp-l4/README.md).

---

## Quick Start (No NVIDIA GPU Required)

Core dependencies are just `usd-core`, `pillow`, `numpy`, `scipy`, and `zxing-cpp`. On macOS or Linux CPU, `uv` runs static stage validation, 4-view Blender Cycles path-traced lookdev, and the 99-test local suite:

```sh
git clone https://github.com/ekusiadadus/issac-harness.git
cd issac-harness
uv sync --locked

# 1. Audit composed USD refs, materials, color spaces, 1/10-unit camera specs, UVs, and support gaps
uv run --locked python -m harness check examples/inspection

# 2. Render nominal and -2 EV lighting cases on CPU (Blender Cycles) and build an HTML review gallery
uv run --locked python -m harness render examples/inspection --capture-only --out out/first-review

# 3. Run the local test suite
uv run --locked python -m pytest -q
```

Open `out/first-review/index.html` to inspect full-resolution overview and close-up PNGs, target-box luminance statistics, and EAN-13 barcode decode results (`0426112500014`).

---

## Silent Failure Modes Caught by This Harness

Every check in `harness/` and rule in [`AGENTS.md`](AGENTS.md) exists because an authoring script or coding agent hit a concrete silent failure mode:

1. **100× lens and filmback typos that preserve field of view**
   OpenUSD (`UsdGeomCamera`) stores `focalLength` and `horizontalAperture` in **tenths of a scene unit** (`0.58` and `0.36` for a 58 mm lens on a 36 mm gate when `metersPerUnit = 1`). Multiplying both by 100 yields the exact same framing in the viewport while breaking depth of field and camera calibration. `harness check` cross-checks USD attributes against declared `focal_length_mm` specs.
2. **A 13,738-pixel segmentation hole missed by cross-checking annotators**
   Toggling a surface mark from hidden to visible mid-run caused **both** semantic and instance segmentation annotators to flip 13,738 pixels of the underlying carton to background (`0`)—even when the mark was converted to an opaque `alpha_cutoff = 128` cutout mesh (13,670 pixels lost). `issac-harness` keeps child marks permanently registered to RTX, parks unused marks behind the active camera, and enforces `off -> on -> off` pixel invariance via `[[appearance_pairs]]` (`campaign label-invariance`).
3. **Black un-tiled UVs and config-level "diversity" illusions**
   `campaign texel-check` computes worst-direction world `texels/m` and checks explicit `wrapS` / `wrapT` repeat modes outside `[0, 1]`. `[min_distinct_texture_sets]` ignores config finish names and counts unique SHA-256 tuples of the base-color, roughness, and normal image files actually bound.
4. **Reset-frame physics impulses caught by negative controls**
   Alongside rigid-body drop settling (`runtime/lab_smoke.py`), kinematic two-finger grasp trials pair closed-finger lifts with an **open-finger negative control** along the identical trajectory—catching articulation resets that collapse fingers through the target object before step 1.
5. **Loud failure on real-camera qualification (`campaign release`)**
   Synthetic holdout splits overestimate transfer. `campaign release` checks cross-process replay determinism, registered real-image RGB-D error, byte-deduplicated real-image detection AP, and human full-resolution sign-off (`review.json` / `review.md`), exiting `1` whenever any pre-committed threshold in `release.toml` is missed.

![Side-by-side 4-frame replay check across two independent Kit processes for the same seed: Run 1 with tight boxes (left), Run 2 with pixel-exact masks (center), and 25x RGB residual heatmap on white (right; normalized sRGB RMSE 0.00324–0.00349, mask diff 0 px)](docs/assets/replay-comparison.gif)

---

## Generate and Qualify a Dataset from Your Own OpenUSD Stage

Start from any metric (`metersPerUnit = 1`), Z-up OpenUSD stage and target prim paths:

```sh
uv run --locked python -m harness campaign init assets/parts.usda \
  --target part=/Factory/PartA --target part=/Factory/PartB --out jobs/parts
uv run --locked python -m harness campaign plan jobs/parts/campaign.toml \
  --frames 50000 --shard-size 128 --seed 42 --out jobs/parts/plan.json
```

On a Linux host with an RT-Core GPU (L4, L40S, or RTX series; A100/H100 lack RT Cores for RTX rendering), [`scripts/isaac_python.sh`](scripts/isaac_python.sh) runs digest-pinned `isaac-sim` (6.2.0) and `isaac-lab` (v3.0.0) NGC containers while preserving host absolute paths and reusing Kit/OptiX shader caches (`HARNESS_KIT_CACHE`):

```sh
export ACCEPT_EULA=Y   # Set only on hosts where you accept NVIDIA's license terms
export HARNESS_KIT_CACHE="$PWD/out/.kit-cache"

uv run --locked python -m harness campaign run-plan jobs/parts/campaign.toml \
  --plan jobs/parts/plan.json --out out/parts --max-shards 1 --asset-root "$PWD"
uv run --locked python -m harness campaign check out/parts \
  --count 128 --coco --out out/parts/labels
uv run --locked python -m harness campaign gallery out/parts \
  --indices 0 17 43 95 --out out/parts/review
```

Because raw Replicator numeric IDs can vary across processes and native tight boxes `[x_min, y_min, x_max, y_max]` use inclusive bounds, `campaign check --coco` and `dataset-check` normalize outputs to deterministic class IDs, RLE instance masks, metric range arrays (`float32` meters), and half-open `[x0, y0, x1, y1)` intervals.

---

## CLI & Runtime Overview

| Command / Script | Runtime | Purpose |
|---|---|---|
| `harness init scenes/my-cell` | CPU | Scaffold a standalone OpenUSD workcell package with `scene.toml`, layers, and CC0 textures |
| `harness check <dir> [--nvidia out]` | CPU | Verify references, material bindings, color spaces, normal decoding, UVs, camera units, resting support gaps, and `usd-validation-nvidia 1.23.0` (49 rules) |
| `harness simready <asset.usdc>` | CPU | Run isolated NVIDIA SimReady Robotics-Prop 5.0.0 validation per prop |
| `harness render --backend cycles\|storm\|rtx` | CPU / RTX | Capture identical views across Cycles, Storm (`usdrecord`), or Isaac Sim RTX Path Tracing |
| `harness accept <review-dir>` | CPU | Verify reference hashes and require written `pass`/`fail` observations for every view in `review.md` |
| `runtime/lab_smoke.py` | Linux RTX | Load `environment.usda` in Isaac Lab v3.0.0 (PhysX) and measure drop distance, resting height error, and settling velocity |
| `harness campaign ...` | CPU / RTX | Plan shards, capture synchronized 4-modal frames, check cross-process replay, project through calibrated TUM lens models, compare registered real RGB-D, evaluate real-image detectors, and gate `release` |

---

## Documentation

- [Authoring & Visual Review Guide](docs/authoring.md)
- [Dataset Campaigns & Qualification](docs/campaigns.md)
- [Localized Surface Marks & Label Invariance](docs/surface-marks.md)
- [World-Space Texel Density Checks](docs/texture-scale.md)
- [Intra-Class Geometry Variants](docs/asset-families.md)
- [Registered Real RGB-D Quality & Exposure Freezing](docs/registered-quality.md)
- [Calibrated Color Sensor Projection](docs/color-sensor.md) · [Sensor Model](docs/sensor-model.md)
- [Photographic HDRI & Environment Lighting](docs/environment-light.md)
- [PhysX Contact, Sliding & Two-Finger Grasp Trials](docs/contact-trials.md)
- [RTX, Replicator & Isaac Lab Runtime Commands](runtime/README.md)
- [Public References & Design Provenance](docs/references.md)

---

## Scope Boundaries & License

- **What this proves (and what it does not)**: `issac-harness` enforces stage integrity, cross-process render/mask determinism, authored-collider PhysX settling, and pre-committed real-image evaluation thresholds. It does not replace on-site hardware measurements for floor friction, motor thermal limits, flexible film vacuum sealing, or safety interlocks.
- **License**: Source code, schemas, and original sample configurations/geometry in this repository are licensed under the [Apache License 2.0](LICENSE). Bundled third-party PBR scans/HDRIs from Poly Haven and ambientCG are CC0, and evaluation artifacts derived from public academic benchmarks (BOP YCB-Video and TUM RGB-D / Mono) remain under their respective upstream terms (see [`examples/inspection/ASSETS.md`](examples/inspection/ASSETS.md) and [`docs/references.md`](docs/references.md)). Running NVIDIA Isaac Sim and Isaac Lab containers requires accepting NVIDIA's applicable license terms.
