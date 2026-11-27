"""Capture named cameras from one local USD with Isaac Sim 6.2 standalone.

Run with Isaac Sim's python.sh, not the system Python. No scene is authored here.
"""

import argparse
import hashlib
import json
import re
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def camera_arg(value):
    name, separator, path = value.partition("=")
    if not separator or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or not path.startswith("/"):
        raise argparse.ArgumentTypeError("expected NAME=/absolute/USD/CameraPath")
    return name, path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path, help="Local composed USD stage")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--camera", action="append", required=True, type=camera_arg, metavar="NAME=PRIM")
    parser.add_argument("--renderer", choices=("PathTracing", "RealTimePathTracing"), default="PathTracing")
    parser.add_argument("--width", type=positive_int, default=1200)
    parser.add_argument("--height", type=positive_int, default=800)
    parser.add_argument(
        "--spp", type=positive_int, default=64, help="PathTracing samples per pixel per frame"
    )
    parser.add_argument("--warmup-frames", type=positive_int, default=32)
    parser.add_argument("--rt-subframes", type=positive_int, default=8)
    parser.add_argument("--max-bounces", type=positive_int, default=4)
    args = parser.parse_args(argv)
    args.scene = args.scene.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.scene.is_file() or args.scene.suffix.lower() not in (".usd", ".usda", ".usdc"):
        parser.error("--scene must be an existing local USD file")
    if len({name for name, _ in args.camera}) != len(args.camera):
        parser.error("camera names must be unique")
    if args.renderer != "PathTracing" and args.spp != 64:
        parser.error("--spp applies only to PathTracing")
    return args


def png_dimensions(path):
    with path.open("rb") as file:
        header = file.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise RuntimeError(f"invalid PNG: {path}")
    return struct.unpack(">II", header[16:24])


def configure_render(settings, args):
    """Apply stable color settings and read back settings that affect RGB."""
    requested = {
        "/rtx/rendermode": args.renderer,
        "/rtx/post/aa/op": 0,  # Avoid NGX/DLSS in headless RTX capture.
        "/rtx/post/histogram/enabled": False,
        "/rtx/post/tonemap/filmIso": 100.0,
        "/rtx/post/tonemap/cameraShutter": 1.0,
        "/rtx/post/tonemap/fNumber": 5.0,
        "/rtx/post/tonemap/op": 1,  # Linear (Off), per RTX operator table.
        "/rtx/post/tonemap/colorMode": 0,  # sRGB linear.
    }
    if args.renderer == "PathTracing":
        requested["/rtx/pathtracing/spp"] = args.spp
        requested["/rtx/pathtracing/maxBounces"] = args.max_bounces
        requested["/rtx/pathtracing/totalSpp"] = args.spp * (
            args.warmup_frames + args.rt_subframes * getattr(args, "frames", 1)
        )
    for key, value in requested.items():
        settings.set(key, value)
    effective = {key: settings.get(key) for key in requested}
    if any(effective[key] != value for key, value in requested.items()):
        raise RuntimeError(f"renderer settings did not apply: {effective}")
    effective["/rtx/pathtracing/optixDenoiser/enabled"] = settings.get(
        "/rtx/pathtracing/optixDenoiser/enabled"
    )
    return effective


def wait_for_annotators(app, rep, annotators, args, *, max_attempts=24, timeout_s=120):
    """Step a paused world until all attached AOVs provide a nonempty frame."""
    import numpy as np

    deadline = time.monotonic() + timeout_s
    shapes = {}
    for attempt in range(1, max_attempts + 1):
        if time.monotonic() >= deadline or not app.is_running():
            break
        app.update()
        rep.orchestrator.step(
            rt_subframes=args.rt_subframes,
            pause_timeline=True,
            delta_time=0.0,
            wait_for_render=True,
        )
        payloads = {name: annotator.get_data() for name, annotator in annotators.items()}
        shapes = {}
        ready = True
        for name, payload in payloads.items():
            data = payload.get("data") if isinstance(payload, dict) else payload
            array = np.asarray(data) if data is not None else np.asarray([])
            shapes[name] = list(array.shape)
            if array.size == 0:
                ready = False
            elif name == "rgb" and (array.ndim != 3 or tuple(array.shape[:2]) != (args.height, args.width)):
                ready = False
            elif name in ("semantic", "depth") and tuple(array.shape[:2]) != (args.height, args.width):
                ready = False
        if ready:
            return payloads, attempt
    raise RuntimeError(
        f"annotators never became ready after {attempt} paused steps "
        f"(timeout={timeout_s}s, shapes={shapes}, renderer={args.renderer})"
    )


def configure_camera_exposure(camera):
    """Author the USD camera controls that supersede global RTX tone-map exposure."""
    requested = {
        "exposure": (camera.GetExposureAttr(), 0.0),
        "exposure:fStop": (camera.GetExposureFStopAttr(), 5.0),
        "exposure:iso": (camera.GetExposureIsoAttr(), 100.0),
        "exposure:responsivity": (camera.GetExposureResponsivityAttr(), 1.0),
        "exposure:time": (camera.GetExposureTimeAttr(), 1.0),
    }
    for name, (attr, value) in requested.items():
        attr.Set(value)
        if float(attr.Get()) != value:
            raise RuntimeError(f"camera exposure did not apply: {name}={attr.Get()}")


def camera_model(camera, width, height, meters_per_unit):
    """Report ideal pinhole intrinsics; no unmeasured lens/noise model is implied."""
    focal = float(camera.GetFocalLengthAttr().Get())
    horizontal = float(camera.GetHorizontalApertureAttr().Get())
    vertical = float(camera.GetVerticalApertureAttr().Get())
    if min(focal, horizontal, vertical) <= 0:
        raise RuntimeError("camera focal length and apertures must be positive")
    h_offset = float(camera.GetHorizontalApertureOffsetAttr().Get())
    v_offset = float(camera.GetVerticalApertureOffsetAttr().Get())
    if abs(h_offset) > 1e-9 or abs(v_offset) > 1e-9:
        raise RuntimeError(
            f"nonzero aperture offset is not supported by pinhole export: {(h_offset, v_offset)}"
        )
    prim = camera.GetPrim()
    authored_distortion = [
        attr.GetName()
        for attr in prim.GetAttributes()
        if ("distortion" in attr.GetName().lower() or "fisheye" in attr.GetName().lower())
        and attr.HasAuthoredValueOpinion()
    ]
    if authored_distortion:
        raise RuntimeError(f"authored lens distortion is not supported: {authored_distortion}")
    from pxr import Usd, UsdGeom

    world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    clip = camera.GetClippingRangeAttr().Get()
    authored_exposure = {
        attr.GetName(): attr.Get()
        for attr in prim.GetAttributes()
        if "exposure" in attr.GetName().lower() and attr.HasAuthoredValueOpinion()
    }

    return {
        "model": "ideal_pinhole",
        "intrinsics_px": {
            "fx": width * focal / horizontal,
            "fy": height * focal / vertical,
            "cx": width / 2,
            "cy": height / 2,
        },
        # USD camera lens values are tenths of a stage unit.
        "focal_length_mm": focal * meters_per_unit * 100,
        "horizontal_aperture_mm": horizontal * meters_per_unit * 100,
        "vertical_aperture_mm": vertical * meters_per_unit * 100,
        "meters_per_unit": meters_per_unit,
        "camera_to_world_stage_units": [[float(world[i][j]) for j in range(4)] for i in range(4)],
        "clipping_range_stage_units": [float(clip[0]), float(clip[1])],
        "aperture_offset": [h_offset, v_offset],
        "authored_camera_exposure": authored_exposure,
        "lens_distortion": "none modeled",
        "sensor_noise": "none modeled",
    }


def gpu_inventory():
    """Capture physical GPU identity separately from the renderer setting."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"source": "nvidia-smi", "available": False}
    return {
        "source": "nvidia-smi",
        "available": result.returncode == 0,
        "rows": result.stdout.strip().splitlines() if result.returncode == 0 else [],
        "error": result.stderr.strip() if result.returncode else "",
    }


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        launch_config={
            "headless": True,
            "renderer": args.renderer,
            "width": args.width,
            "height": args.height,
            "samples_per_pixel_per_frame": args.spp,
            "max_bounces": args.max_bounces,
            "denoiser": True,
            "anti_aliasing": 0,
            "multi_gpu": False,
            "sync_loads": True,
        }
    )
    try:
        capture(args, simulation_app)
    except BaseException:
        traceback.print_exc()
        simulation_app.close(exit_code=1)
        raise
    simulation_app.close()
    return 0


def renderer_versions():
    import omni.kit.app
    from isaacsim.core.version import get_version

    manager = omni.kit.app.get_app().get_extension_manager()
    extension = manager.get_enabled_extension_id("omni.replicator.core")
    if not extension:
        raise RuntimeError("could not identify the active Replicator extension")
    return {"isaacsim_app": list(get_version()), "replicator_extension": extension}


def capture(args, app):
    import carb.settings
    import numpy as np
    import omni.replicator.core as rep
    from omni.replicator.core.functional import write_image
    from isaacsim.core.experimental.utils.stage import open_stage
    from pxr import UsdGeom

    opened, stage = open_stage(str(args.scene))
    if not opened or stage is None:
        raise RuntimeError(f"Isaac Sim could not open {args.scene}")
    app.reset_render_settings()
    layers = []
    for layer in stage.GetUsedLayers():
        identifier = layer.identifier
        if "://" in identifier or identifier.startswith("omniverse:"):
            raise RuntimeError(f"remote USD layer is not allowed for offline capture: {identifier}")
        if layer.realPath:
            layer_path = Path(layer.realPath)
            if not layer_path.is_file():
                raise RuntimeError(f"USD layer is not a local file: {identifier}")
            layers.append(
                {"path": str(layer_path), "sha256": hashlib.sha256(layer_path.read_bytes()).hexdigest()}
            )
    for name, path in args.camera:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsActive() or not prim.IsA(UsdGeom.Camera):
            raise RuntimeError(f"missing or non-camera prim for {name}: {path}")
        configure_camera_exposure(UsdGeom.Camera(prim))

    settings = carb.settings.get_settings()
    settings.set("/app/player/playSimulations", False)
    effective_settings = configure_render(settings, args)
    rep.orchestrator.set_capture_on_play(False)
    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()
    timeline.commit()

    captures = []
    for name, camera_path in args.camera:
        camera_dir = args.output_dir / name
        camera_dir.mkdir(exist_ok=True)
        rp = rep.create.render_product(camera_path, (args.width, args.height), name=f"{name}_capture")
        annotator = rep.annotators.get("rgb")
        annotator_attached = False
        try:
            annotator.attach(rp)
            annotator_attached = True
            for _ in range(args.warmup_frames):
                if not app.is_running():
                    raise RuntimeError(f"Isaac Sim closed during {name} warmup")
                app.update()
            payloads, readiness_steps = wait_for_annotators(app, rep, {"rgb": annotator}, args)
            data = payloads["rgb"]
            if data is None or getattr(data, "shape", None) is None:
                raise RuntimeError(f"no RGB frame for {name}")
            if (
                len(data.shape) != 3
                or tuple(data.shape[:2]) != (args.height, args.width)
                or data.shape[2] not in (3, 4)
            ):
                raise RuntimeError(f"unexpected RGB shape for {name}: {data.shape}")
            rgb = np.asarray(data[:, :, :3])
            if not np.any(rgb != rgb[0, 0, :]) or (
                data.shape[2] == 4 and not np.asarray(data[:, :, 3]).any()
            ):
                raise RuntimeError(f"uniform or transparent RGB frame for {name}")
            write_image(path=str(camera_dir / "rgb.png"), data=data)
        finally:
            if annotator_attached:
                annotator.detach()
            rp.destroy()
        pngs = list(camera_dir.rglob("*.png"))
        if len(pngs) != 1 or png_dimensions(pngs[0]) != (args.width, args.height):
            raise RuntimeError(f"expected one {args.width}x{args.height} PNG for {name}, found {pngs}")
        captures.append(
            {
                "name": name,
                "camera_prim": camera_path,
                "png": str(pngs[0].relative_to(args.output_dir)),
                "sha256": hashlib.sha256(pngs[0].read_bytes()).hexdigest(),
                "rgb_shape": list(data.shape),
                "readiness_steps": readiness_steps,
                "camera_model": camera_model(
                    UsdGeom.Camera(stage.GetPrimAtPath(camera_path)),
                    args.width,
                    args.height,
                    UsdGeom.GetStageMetersPerUnit(stage),
                ),
            }
        )

    manifest = {
        "status": "captured",
        "scene": str(args.scene),
        "scene_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
        "usd_layers": layers,
        "versions": renderer_versions(),
        "render": {
            "effective_settings": effective_settings,
            "resolution": [args.width, args.height],
            "requested_launch": {
                "renderer": args.renderer,
                "max_bounces": args.max_bounces,
                "denoiser": True,
                "multi_gpu": False,
            },
            "warmup_frames": args.warmup_frames,
            "rt_subframes": args.rt_subframes,
            "capture_on_play": False,
            "configured_engine": "RTX"
            if effective_settings["/rtx/rendermode"] in ("PathTracing", "RealTimePathTracing")
            else "unknown",
            "active_gpu_index_setting": settings.get("/renderer/activeGpu"),
            "gpu_inventory": gpu_inventory(),
        },
        "captures": captures,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    sys.exit(main())
