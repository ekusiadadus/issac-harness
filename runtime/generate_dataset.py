"""Generate a small, fully annotated RTX pilot from the inspection USD.

Run with Isaac Sim 6.2 python.sh. Each numbered frame is an explicit paused
Replicator step; metadata records the sampled and read-back scene state.
"""

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from capture_isaac import (
    renderer_versions,
    camera_model,
    configure_camera_exposure,
    configure_render,
    gpu_inventory,
    positive_int,
    wait_for_annotators,
)


OBJECTS = {
    "carton": "/World/Props/Carton",
    "handheld_scanner": "/World/Props/Handset",
}
TARGET_MESHES = {
    "carton": "/World/Props/Carton/Base",
    "handheld_scanner": "/World/Props/Handset/Body",
}
FINISHES = {
    "carton": ("standard", "worn"),
    "handheld_scanner": ("standard", "matte"),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera", default="/World/Cameras/Overview")
    parser.add_argument("--frames", type=positive_int, default=4)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--width", type=positive_int, default=768)
    parser.add_argument("--height", type=positive_int, default=512)
    parser.add_argument("--renderer", choices=("PathTracing", "RealTimePathTracing"), default="PathTracing")
    parser.add_argument("--spp", type=positive_int, default=32)
    parser.add_argument("--warmup-frames", type=positive_int, default=16)
    parser.add_argument("--rt-subframes", type=positive_int, default=8)
    parser.add_argument("--max-bounces", type=positive_int, default=4)
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    args.scene = args.scene.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.scene.is_file() or args.scene.suffix.lower() not in (".usd", ".usda", ".usdc"):
        parser.error("--scene must be an existing local USD file")
    return args


def file_record(path, root):
    return {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def source_files(stage):
    """Fingerprint composed USD layers and all resolved asset inputs."""
    from pxr import Sdf

    paths = set()
    for layer in stage.GetUsedLayers():
        if layer.anonymous:
            continue
        if not layer.realPath:
            raise RuntimeError(f"USD layer has no local file: {layer.identifier}")
        paths.add(Path(layer.realPath))
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            asset = attr.Get()
            if not isinstance(asset, Sdf.AssetPath) or not asset.path:
                continue
            if not asset.resolvedPath:
                raise RuntimeError(f"unresolved asset at {prim.GetPath()}.{attr.GetName()}: {asset.path}")
            paths.add(Path(asset.resolvedPath))
    result = []
    for path in sorted(paths):
        if not path.is_file():
            raise RuntimeError(f"source file missing: {path}")
        result.append(
            {
                "path": os.path.relpath(path, Path(stage.GetRootLayer().realPath).parent),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return result


def labels_for_id(mapping, semantic_id):
    value = mapping.get(str(int(semantic_id)), mapping.get(int(semantic_id), {}))
    if isinstance(value, dict):
        value = value.get("class", value.get("class:*", ""))
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value)


def validate_frame(rgb, segmentation, boxes, depth, segmentation_mapping, bbox_mapping, width, height):
    """Reject missing data and annotations that cannot align to pixel coordinates.

    Native Replicator x_max/y_max are inclusive pixel coordinates, including
    one-pixel boxes and boxes ending at the image's last row or column.
    """
    if rgb.shape not in ((height, width, 3), (height, width, 4)) or rgb.dtype != np.uint8:
        raise RuntimeError(f"invalid RGB shape/dtype: {rgb.shape}, {rgb.dtype}")
    if not np.any(rgb[:, :, :3] != rgb[0, 0, :3]):
        raise RuntimeError("uniform RGB frame")
    if segmentation.shape != (height, width) or not np.issubdtype(segmentation.dtype, np.integer):
        raise RuntimeError(f"invalid segmentation shape/dtype: {segmentation.shape}, {segmentation.dtype}")
    if depth.shape != (height, width) or not np.issubdtype(depth.dtype, np.floating):
        raise RuntimeError(f"invalid depth shape/dtype: {depth.shape}, {depth.dtype}")
    if not np.any(np.isfinite(depth) & (depth > 0)):
        raise RuntimeError("empty camera depth")
    if not hasattr(boxes, "dtype") or not boxes.dtype.names:
        raise RuntimeError("native tight bounding boxes missing")
    required = {"semanticId", "x_min", "y_min", "x_max", "y_max"}
    if not required.issubset(boxes.dtype.names):
        raise RuntimeError(f"unexpected bbox fields: {boxes.dtype.names}")
    result = []
    for box in boxes.reshape(-1):
        semantic_id = int(box["semanticId"])
        label = labels_for_id(bbox_mapping, semantic_id)
        if label not in OBJECTS:
            continue
        x0, y0, x1, y1 = (int(box[key]) for key in ("x_min", "y_min", "x_max", "y_max"))
        if not (0 <= x0 <= x1 < width and 0 <= y0 <= y1 < height):
            raise RuntimeError(f"invalid {label} bbox: {(x0, y0, x1, y1)}")
        mask_ids = [
            int(candidate)
            for candidate in np.unique(segmentation)
            if labels_for_id(segmentation_mapping, candidate) == label
        ]
        mask = np.isin(segmentation, mask_ids)
        if not mask.any() or not mask[y0:y1 + 1, x0:x1 + 1].any():
            raise RuntimeError(
                f"bbox and mask do not overlap for {label} bbox_id={semantic_id}, segmentation_ids={mask_ids}"
            )
        ys, xs = np.nonzero(mask)
        if xs.min() < x0 - 2 or xs.max() >= x1 + 2 or ys.min() < y0 - 2 or ys.max() >= y1 + 2:
            raise RuntimeError(f"mask extends outside bbox for {label} id={semantic_id}")
        target_depth = depth[mask]
        if not np.any(np.isfinite(target_depth) & (target_depth > 0)):
            raise RuntimeError(f"no finite positive range depth for {label}")
        result.append(
            {
                "class": label,
                "bbox_semantic_id": semantic_id,
                "segmentation_ids": mask_ids,
                "xyxy_px": [x0, y0, x1, y1],
                "visible_pixels": int(mask.sum()),
                "range_depth_m_median": float(
                    np.median(target_depth[np.isfinite(target_depth) & (target_depth > 0)])
                ),
                "occlusion_ratio": float(box["occlusionRatio"])
                if "occlusionRatio" in boxes.dtype.names
                else None,
            }
        )
    missing = set(OBJECTS) - {entry["class"] for entry in result}
    if missing:
        raise RuntimeError(
            f"missing visible native labels/bboxes: {sorted(missing)}; bbox_mapping={bbox_mapping}; segmentation_mapping={segmentation_mapping}"
        )
    return result


def sample_frame(rng):
    return {
        "carton": {
            "xy_offset_m": rng.uniform(-0.025, 0.025, 2).tolist(),
            "yaw_deg": float(rng.uniform(-8, 8)),
            "roughness_multiplier": float(rng.uniform(0.85, 1.1)),
            "surface_finish": str(rng.choice(FINISHES["carton"])),
        },
        "handheld_scanner": {
            "xy_offset_m": rng.uniform(-0.008, 0.008, 2).tolist(),
            "yaw_deg": float(rng.uniform(-8, 8)),
            "roughness_multiplier": float(rng.uniform(0.85, 1.1)),
            "surface_finish": str(rng.choice(FINISHES["handheld_scanner"])),
        },
        "light_multiplier": float(rng.uniform(0.8, 1.2)),
        "tape_visible": bool(rng.integers(0, 2)),
        "back_flap_angle_deg": float(rng.uniform(-7, 25)),
    }


def resolve_roughness_shader(stage, label):
    """Follow the actual mesh binding, rejecting similarly named unbound copies."""
    from pxr import UsdShade

    mesh = stage.GetPrimAtPath(TARGET_MESHES[label])
    if not mesh.IsValid():
        raise RuntimeError(f"missing target mesh: {TARGET_MESHES[label]}")
    material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
    if not material or not material.GetPrim().IsValid():
        raise RuntimeError(f"no material bound to {TARGET_MESHES[label]}")
    material_path = str(material.GetPath())
    if not material_path.startswith(OBJECTS[label] + "/Materials/"):
        raise RuntimeError(f"bound material for {label} is outside object scope: {material_path}")
    shader_path = str(material.GetPath().AppendChild("Roughness"))
    shader = UsdShade.Shader(stage.GetPrimAtPath(shader_path))
    if not shader or not shader.GetPrim().IsValid():
        raise RuntimeError(f"bound roughness shader missing: {shader_path}")
    return shader_path


def set_roughness(stage, path, multiplier):
    from pxr import Gf, Sdf, UsdShade

    shader = UsdShade.Shader(stage.GetPrimAtPath(path))
    if not shader or not shader.GetPrim().IsValid():
        raise RuntimeError(f"missing roughness texture shader: {path}")
    input_scale = shader.GetInput("scale") or shader.CreateInput("scale", Sdf.ValueTypeNames.Float4)
    baseline = input_scale.Get() or Gf.Vec4f(1, 1, 1, 1)
    input_scale.Set(
        Gf.Vec4f(*(float(baseline[i]) * multiplier if i < 3 else float(baseline[i]) for i in range(4)))
    )
    return input_scale, {
        "variant_base_scale": list(baseline),
        "effective_scale": list(input_scale.Get()),
        "bias": list(shader.GetInput("bias").Get()) if shader.GetInput("bias") else None,
        "shader_path": path,
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
        generate(args, simulation_app)
    except BaseException:
        traceback.print_exc()
        simulation_app.close(exit_code=1)
        raise
    simulation_app.close()
    return 0


def generate(args, app):
    import carb.settings
    import omni.replicator.core as rep
    from isaacsim.core.experimental.utils.stage import open_stage
    from omni.replicator.core.functional import write_image
    from pxr import Gf, UsdGeom, UsdLux

    opened, stage = open_stage(str(args.scene))
    if not opened or stage is None:
        raise RuntimeError(f"could not open {args.scene}")
    app.reset_render_settings()
    camera_prim = stage.GetPrimAtPath(args.camera)
    if not camera_prim.IsA(UsdGeom.Camera):
        raise RuntimeError(f"missing camera: {args.camera}")
    configure_camera_exposure(UsdGeom.Camera(camera_prim))
    original_transforms = {}
    variants = {}
    for label, path in OBJECTS.items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"missing object {label}: {path}")
        xform = UsdGeom.Xformable(prim)
        translate = next(
            (op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None
        )
        rotate = next(
            (op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeRotateZ), None
        )
        if translate is None:
            raise RuntimeError(f"{path} needs a translate xform op for supported variation")
        if rotate is None:
            rotate = xform.AddRotateZOp()
            rotate.Set(0.0)
        original_transforms[label] = (translate, rotate, tuple(translate.Get()), float(rotate.Get()))
        variant = prim.GetVariantSet("surface_finish")
        if not variant.IsValid() or not set(FINISHES[label]).issubset(set(variant.GetVariantNames())):
            raise RuntimeError(f"missing surface_finish variants on {path}: {FINISHES[label]}")
        variants[label] = (variant, variant.GetVariantSelection())
    lights = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdLux.LightAPI):
            light = UsdLux.LightAPI(prim)
            intensity = light.GetIntensityAttr()
            if intensity and intensity.HasAuthoredValueOpinion():
                lights.append((str(prim.GetPath()), intensity, float(intensity.Get())))
    if not lights:
        raise RuntimeError("no authored light intensity to vary")
    flap = UsdGeom.Mesh.Get(stage, "/World/Props/Carton/BackFlap")
    if not flap:
        raise RuntimeError("missing carton back flap for geometry randomization")
    lower, upper = flap.GetExtentAttr().Get()
    hinge = Gf.Vec3d((lower[0] + upper[0]) / 2, lower[1], (lower[2] + upper[2]) / 2)
    flap.AddTranslateOp(opSuffix="datasetHinge").Set(hinge)
    flap_angle = flap.AddRotateXOp(opSuffix="datasetFold")
    flap.AddTranslateOp(opSuffix="datasetHinge", isInverseOp=True)
    tape = UsdGeom.Imageable(stage.GetPrimAtPath("/World/Props/Carton/TapeSeam"))
    if not tape or not tape.GetPrim().IsValid():
        raise RuntimeError("missing Carton/TapeSeam geometry choice")
    sources = source_files(stage)
    settings = carb.settings.get_settings()
    settings.set("/app/player/playSimulations", False)
    render_settings = configure_render(settings, args)
    rep.set_global_seed(args.seed)
    replicator_seed = int(rep.get_global_seed())
    if replicator_seed != args.seed:
        raise RuntimeError(f"Replicator seed readback mismatch: {replicator_seed}")
    rep.orchestrator.set_capture_on_play(False)
    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()
    timeline.commit()
    rp = rep.create.render_product(args.camera, (args.width, args.height), name="dataset_capture")
    annotators = {
        "rgb": rep.annotators.get("rgb"),
        "semantic": rep.annotators.get(
            "semantic_segmentation", init_params={"colorize": False, "semanticTypes": ["class"]}
        ),
        "bbox": rep.annotators.get("bounding_box_2d_tight", init_params={"semanticTypes": ["class"]}),
        "depth": rep.annotators.get("distance_to_camera"),
    }
    for annotator in annotators.values():
        annotator.attach(rp)
    frames = []
    overridden_scales = []
    rng = np.random.default_rng(args.seed)
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    try:
        for _ in range(args.warmup_frames):
            if not app.is_running():
                raise RuntimeError("Isaac Sim stopped during warmup")
            app.update()
        for index in range(args.frames):
            for input_scale in overridden_scales:
                input_scale.GetAttr().Clear()
            overridden_scales.clear()
            sampled = sample_frame(rng)
            readback = {"objects": {}, "lights": {}}
            for label, (translate, rotate, base_xyz, base_yaw) in original_transforms.items():
                sample = sampled[label]
                variant = variants[label][0]
                variant.SetVariantSelection(sample["surface_finish"])
                if variant.GetVariantSelection() != sample["surface_finish"]:
                    raise RuntimeError(f"surface_finish did not apply for {label}")
                translate.Set(
                    Gf.Vec3d(
                        base_xyz[0] + sample["xy_offset_m"][0] / meters_per_unit,
                        base_xyz[1] + sample["xy_offset_m"][1] / meters_per_unit,
                        base_xyz[2],
                    )
                )
                rotate.Set(base_yaw + sample["yaw_deg"])
                shader_path = resolve_roughness_shader(stage, label)
                scale_input, roughness_readback = set_roughness(
                    stage, shader_path, sample["roughness_multiplier"]
                )
                overridden_scales.append(scale_input)
                readback["objects"][label] = {
                    "translate_stage_units": list(translate.Get()),
                    "rotate_z_deg": float(rotate.Get()),
                    "surface_finish": variant.GetVariantSelection(),
                    "roughness": roughness_readback,
                }
            for path, attr, base in lights:
                attr.Set(base * sampled["light_multiplier"])
                readback["lights"][path] = float(attr.Get())
            flap_angle.Set(sampled["back_flap_angle_deg"])
            readback["back_flap"] = {"angle_deg": flap_angle.Get(), "hinge_local_stage_units": list(hinge)}
            tape.GetVisibilityAttr().Set("inherited" if sampled["tape_visible"] else "invisible")
            readback["tape_visibility"] = tape.GetVisibilityAttr().Get()
            settings.set("/rtx-transient/post/dlss/forceParamReset", True)
            payloads, readiness_steps = wait_for_annotators(app, rep, annotators, args)
            rgb = np.asarray(payloads["rgb"])
            sem_payload = payloads["semantic"]
            box_payload = payloads["bbox"]
            depth = np.asarray(payloads["depth"], dtype=np.float32).squeeze()
            segmentation = np.asarray(sem_payload["data"]).squeeze()
            mapping = sem_payload["info"]["idToLabels"]
            bbox_mapping = box_payload["info"]["idToLabels"]
            boxes = np.asarray(box_payload["data"])
            checked = validate_frame(
                rgb, segmentation, boxes, depth, mapping, bbox_mapping, args.width, args.height
            )
            frame_dir = args.output_dir / f"frame_{index:04d}"
            frame_dir.mkdir()
            rgb_path = frame_dir / "rgb.png"
            sem_path = frame_dir / "semantic_ids.npy"
            depth_path = frame_dir / "distance_to_camera_m.npy"
            write_image(path=str(rgb_path), data=rgb)
            np.save(sem_path, segmentation.astype(np.uint32))
            np.save(depth_path, depth)
            frame = {
                "index": index,
                "sampled": sampled,
                "readback": readback,
                "readiness_steps": readiness_steps,
                "boxes": checked,
                "segmentation_id_to_labels": {str(k): v for k, v in mapping.items()},
                "bbox_id_to_labels": {str(k): v for k, v in bbox_mapping.items()},
                "files": [file_record(path, args.output_dir) for path in (rgb_path, sem_path, depth_path)],
            }
            (frame_dir / "annotation.json").write_text(json.dumps(frame, indent=2) + "\n")
            frame["files"].append(file_record(frame_dir / "annotation.json", args.output_dir))
            frames.append(frame)
    finally:
        for input_scale in overridden_scales:
            input_scale.GetAttr().Clear()
        for label, (variant, original) in variants.items():
            variant.SetVariantSelection(original)
        for annotator in annotators.values():
            annotator.detach()
        rp.destroy()
    rep.orchestrator.wait_until_complete()
    manifest = {
        "status": "captured",
        "scene": str(args.scene),
        "scene_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
        "source_files": sources,
        "versions": renderer_versions(),
        "seed": args.seed,
        "replicator_seed": replicator_seed,
        "numpy_rng": type(rng.bit_generator).__name__,
        "frames_requested": args.frames,
        "camera_prim": args.camera,
        "depth": {
            "annotator": "distance_to_camera",
            "meaning": "Euclidean camera-origin range",
            "unit": "meters",
            "missing_value": 0,
        },
        "camera_model": camera_model(UsdGeom.Camera(camera_prim), args.width, args.height, meters_per_unit),
        "render": {
            "effective_settings": render_settings,
            "resolution": [args.width, args.height],
            "warmup_frames": args.warmup_frames,
            "rt_subframes": args.rt_subframes,
            "configured_engine": "RTX"
            if render_settings["/rtx/rendermode"] in ("PathTracing", "RealTimePathTracing")
            else "unknown",
            "active_gpu_index_setting": settings.get("/renderer/activeGpu"),
            "gpu_inventory": gpu_inventory(),
            "capture_on_play": False,
            "frame_step_delta_time": 0.0,
            "exposure_scope": "USD camera exposure read back; RTX global post settings also read back; no sensor response/noise calibration",
        },
        "frames": frames,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    sys.exit(main())
