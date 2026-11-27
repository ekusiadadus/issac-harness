"""Render all selected case/view pairs and retain their inputs, pixels and settings."""

import json
import math
import shutil
import subprocess
from datetime import UTC, datetime
from html import escape
from pathlib import Path

import zxingcpp
from PIL import Image, ImageStat

from .project import file_hash, write_json
from .review import template

ROOT = Path(__file__).resolve().parents[1]


def run_logged(command, log):
    with log.open("a", encoding="utf-8") as stream:
        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if process.returncode:
        raise RuntimeError(f"Renderer exited {process.returncode}. See {log}")


def render_case(backend, directory, stage, views, samples, isaac_python):
    log = directory / "render.log"
    if backend == "cycles":
        job = directory / "job.json"
        write_json(job, {"stage": str(stage), "views": views, "samples": samples})
        run_logged(["uv", "run", "--script", str(ROOT / "scripts/preview_cycles.py"), str(job)], log)
        job.unlink()
    elif backend == "storm":
        executable = shutil.which("usdrecord")
        if not executable:
            raise ValueError("Storm needs usdrecord on PATH. Use --backend cycles for CPU path tracing.")
        help_text = subprocess.run([executable, "--help"], capture_output=True, text=True, check=True)
        default_time = ["--defaultTime"]
        if "--defaultTime BOOLEAN" in help_text.stdout:
            default_time.append("true")  # Apple USD tools use an explicit boolean.
        for name, view in views.items():
            run_logged(
                [
                    executable,
                    "--renderer",
                    "Storm",
                    "--disableCameraLight",
                    *default_time,
                    "--colorCorrectionMode",
                    "sRGB",
                    "--imageWidth",
                    str(view["size"][0]),
                    "--camera",
                    view["camera"],
                    str(stage),
                    str(directory / (name + ".png")),
                ],
                log,
            )
        version = subprocess.run([executable, "--version"], capture_output=True, text=True, check=False)
        write_json(
            directory / "renderer.json",
            {
                "renderer": "Hydra Storm",
                "camera_light": False,
                "color_correction": "sRGB",
                "version": (version.stdout or version.stderr).strip(),
            },
        )
    else:
        if not isaac_python:
            raise ValueError(
                "Supply --isaac-python pointing to Isaac Sim 6.2 python.sh or its Python interpreter"
            )
        # A Replicator render product has a fixed size. Batch cameras sharing the same filmback ratio.
        sizes = sorted({tuple(v["size"]) for v in views.values()})
        manifests = []
        for width, height in sizes:
            group = directory / f"replicator-{width}x{height}"
            args = []
            for name, view in views.items():
                if view["size"] == [width, height]:
                    args.extend(["--camera", f"{name}={view['camera']}"])
            run_logged(
                [
                    str(isaac_python),
                    str(ROOT / "runtime/capture_isaac.py"),
                    "--scene",
                    str(stage),
                    "--output-dir",
                    str(group),
                    "--renderer",
                    "PathTracing",
                    "--width",
                    str(width),
                    "--height",
                    str(height),
                    "--spp",
                    str(samples),
                    *args,
                ],
                log,
            )
            manifest = json.loads((group / "manifest.json").read_text())
            for capture in manifest["captures"]:
                shutil.copyfile(group / capture["png"], directory / (capture["name"] + ".png"))
            manifests.append(str((group / "manifest.json").relative_to(directory)))
        write_json(
            directory / "renderer.json", {"renderer": "Isaac Sim RTX PathTracing", "manifests": manifests}
        )
    return json.loads((directory / "renderer.json").read_text())


def inspect_frame(path, expected, auto_height=False, target_rect=None, barcode=None):
    with Image.open(path) as im:
        # usdrecord computes its own integer height from a floating-point camera aspect.
        if im.format != "PNG" or im.width != expected[0] or abs(im.height - expected[1]) > int(auto_height):
            raise ValueError(
                f"{path}: expected a {expected[0]} × {expected[1]} PNG; got {im.format} {im.size}"
            )
        if "A" in im.getbands() and im.getchannel("A").getextrema()[1] == 0:
            raise ValueError(f"{path}: fully transparent frame")
        rgb = im.convert("RGB")
        if all(lo == hi for lo, hi in rgb.getextrema()):
            raise ValueError(f"{path}: constant RGB frame")
        luma = rgb.convert("L")
        hist = luma.histogram()
        count = im.width * im.height
        result = {
            "size": list(im.size),
            "sha256": file_hash(path),
            "mean_luma_8bit": round(ImageStat.Stat(luma).mean[0], 2),
            "near_black_pct": round(100 * sum(hist[:3]) / count, 3),
            "near_white_pct": round(100 * sum(hist[-3:]) / count, 3),
        }
        if target_rect:
            left, top, right, bottom = target_rect
            rect = (
                math.floor(left * im.width),
                math.floor(top * im.height),
                math.ceil(right * im.width),
                math.ceil(bottom * im.height),
            )
            crop = rgb.crop(rect)
            result["target_rect_px"] = rect
            result["target_mean_luma_8bit"] = round(ImageStat.Stat(crop.convert("L")).mean[0], 2)
            if barcode:
                reads = [b.text for b in zxingcpp.read_barcodes(crop) if b.valid]
                result["barcode"] = {"expected": barcode, "decoded": reads, "matches": barcode in reads}
        return result


def gallery(project, directory, run):
    sections = []
    for name, view in project.views.items():
        cards = []
        if view.reference:
            ref = run["references"][name]["file"]
            cards.append(
                f'<figure><a href="{escape(ref)}"><img src="{escape(ref)}" alt="{name} reference"></a><figcaption>実写の参照写真</figcaption></figure>'
            )
        for case_name, case in run["cases"].items():
            frame = case["frames"][name]
            src = f"{case_name}/{name}.png"
            read = frame.get("barcode")
            read_text = (" · " + escape(", ".join(read["decoded"]) or "読取不可")) if read else ""
            caption = (
                f"{escape(case_name)} · {frame['size'][0]} × {frame['size'][1]} · "
                f"対象輝度 {frame['target_mean_luma_8bit']:.1f}{read_text}"
            )
            cards.append(
                f'<figure><a href="{src}"><img src="{src}" alt="{escape(name)} / {escape(case_name)}"></a>'
                f"<figcaption>{caption}</figcaption></figure>"
            )
        sections.append(
            f"<section><h2>{escape(name)}</h2><p>{escape(view.inspect)}</p>"
            f'<div class="frames">{"".join(cards)}</div></section>'
        )
    engine = escape(next(iter(run["cases"].values()))["renderer"]["renderer"])
    title = escape(project.name)
    html = f"""<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title} | 撮影比較</title>
<style>
* {{box-sizing:border-box}} body {{margin:0 auto; max-width:1540px; padding:3vw 4vw;
background:#f5f4ef; color:#252a29; font:16px/1.55 Inter,system-ui,sans-serif}}
h1,h2 {{margin:0 0 .3rem}} p {{margin:.2rem 0 1rem}} header,section {{padding:1rem 0;border-bottom:1px solid #c8c9c0}}
.frames {{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));gap:1rem}}
figure {{margin:0}} img {{display:block;width:100%;height:auto;background:#252a29}}
figcaption,small {{font-size:.86rem}} a {{color:#145a49}} </style>
<header><h1>{title}</h1><small>{engine} · <a href="run.json">撮影記録</a> · <a href="check.json">USD 検査</a> · <a href="review.md">目視レビュー</a></small></header>
{"".join(sections)}
<p><small>対象輝度は検査対象を囲む矩形内の PNG 平均輝度（0–255）。画像を開くと原寸で確認できます。</small></p></html>"""
    (directory / "index.html").write_text(html, encoding="utf-8")


def render(project, report, directory, backend, samples, width, isaac_python):
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "check.json", report)
    run = {
        "project": project.name,
        "backend": backend,
        "started_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "cases": {},
    }
    if (ROOT / ".git").exists():
        run["code_commit"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        run["tracked_changes"] = bool(subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip())
    write_json(directory / "run.json", run)
    try:
        if report["status"] != "pass":
            raise ValueError(f"Scene checks failed. See {directory / 'check.json'}")
        for name, case in report["cases"].items():
            case_dir = directory / name
            case_dir.mkdir()
            views = {}
            for key, view in case["views"].items():
                w = width or project.width
                h = max(1, round(w * view["filmback_mm"][1] / view["filmback_mm"][0]))
                views[key] = {"camera": view["camera"], "size": [w, h]}
            renderer = render_case(backend, case_dir, project.cases[name], views, samples, isaac_python)
            frames = {
                key: inspect_frame(
                    case_dir / (key + ".png"),
                    view["size"],
                    auto_height=backend == "storm",
                    target_rect=case["views"][key]["target_rect"],
                    barcode=project.views[key].expected_barcode,
                )
                for key, view in views.items()
            }
            run["cases"][name] = {"renderer": renderer, "frames": frames}
        run["references"] = {}
        for key, view in project.views.items():
            if view.reference:
                name = f"reference-{key}{view.reference.suffix.lower()}"
                shutil.copyfile(view.reference, directory / name)
                run["references"][key] = {"file": name, "sha256": file_hash(view.reference)}
        gallery(project, directory, run)
        template(directory, project, run)
        failed_reads = []
        for name, case in run["cases"].items():
            for key, frame in case["frames"].items():
                if frame.get("barcode", {}).get("matches") is False:
                    failed_reads.append(f"{name}/{key}")
        if failed_reads:
            raise ValueError(
                f"Target barcode unreadable or incorrect in {', '.join(failed_reads)}. See {directory / 'index.html'}"
            )
        run["status"] = "rendered"
    except (Exception, KeyboardInterrupt) as exc:
        run.update(status="failed", error=str(exc) or "interrupted")
        raise
    finally:
        write_json(directory / "run.json", run)
    return directory / "index.html"
