"""Render a configured OpenUSD campaign with synchronized native RTX annotators."""

import argparse
import fcntl
import hashlib
import json
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture_isaac import (
    configure_camera_exposure,
    configure_render,
    positive_int,
    renderer_versions,
)
from generate_dataset import source_files

from harness.campaign import (
    digest,
    inspect_frame,
    load,
    prescribed,
    read,
    sample,
    texture_registry,
    write,
)
from harness.visibility import DescendantParking, depth_witness


def args_for_run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=positive_int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--renderer",
        choices=["PathTracing", "RealTimePathTracing"],
        default="PathTracing",
    )
    parser.add_argument("--spp", type=positive_int, default=16)
    parser.add_argument("--max-bounces", type=positive_int, default=4)
    parser.add_argument("--rt-subframes", type=positive_int, default=4)
    parser.add_argument("--warmup-frames", type=positive_int, default=24)
    args = parser.parse_args()
    if min(args.seed, args.start) < 0:
        parser.error("seed and start must be nonnegative")
    return args


def main():
    args = args_for_run()
    root, cfg = load(args.config)
    cases = prescribed(root, cfg)
    if cases is not None and args.start + args.count > len(cases):
        raise SystemExit("requested capture exceeds number of prescribed views")
    args.width, args.height = cfg["width"], cfg["height"]
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
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
        run(simulation_app, args, root, cfg, cases)
    except BaseException:
        traceback.print_exc()
        simulation_app.close(exit_code=1)
        raise
    simulation_app.close()


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        yield


def set_header(directory, data):
    with lock(directory / ".locks" / "header"):
        target = directory / "campaign.json"
        if target.exists():
            if read(target) != data:
                raise ValueError(
                    f"output belongs to a different configuration: {target}"
                )
        else:
            with NamedTemporaryFile(dir=directory, delete=False) as tmp:
                name = Path(tmp.name)
            write(name, data)
            name.chmod(0o644)
            name.replace(target)


def matrix(rows):
    from pxr import Gf

    return Gf.Matrix4d(*np.array(rows).flatten().tolist())


def rows(transform):
    return np.array(transform).tolist()


def bind_targets(stage, root, cfg):
    from pxr import Gf, Sdf, Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    targets = {}
    for index, spec in enumerate(cfg["instances"], 1):
        if "asset" in spec:
            prim = stage.DefinePrim(spec["prim"], "Xform")
            if not prim.GetReferences().AddReference(
                str((root / spec["asset"]).resolve())
            ):
                raise ValueError(f"reference failed: {spec['asset']}")
        else:
            prim = stage.GetPrimAtPath(spec["prim"])
        if not prim or not prim.IsA(UsdGeom.Xformable):
            raise ValueError(f"expected transformable target {spec['prim']}")
        xf = UsdGeom.Xformable(prim)
        world = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        world.SetTranslateOnly(Gf.Vec3d(0))
        bounds = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        if bounds.IsEmpty():
            raise ValueError(f"target has no extent: {spec['prim']}")
        op = xf.MakeMatrixXform()
        # Use a unique session-only token so overlapping classes retain instance identity.
        token = f"capture{index:05d}"
        prim.AddAppliedSchema("SemanticsLabelsAPI:class")
        prim.CreateAttribute(
            "semantics:labels:class", Sdf.ValueTypeNames.TokenArray
        ).Set([token])
        targets[spec["id"]] = (
            prim,
            op,
            world,
            [bounds.GetCorner(i) for i in range(8)],
            token,
        )
    return targets


def local_surface_probes(targets):
    from pxr import Usd, UsdGeom

    probes = {}
    for name, (root, *_) in targets.items():
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        back = np.asarray(cache.GetLocalToWorldTransform(root).GetInverse())
        surfaces = []
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            coords = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            if not coords.size:
                continue
            coords = coords[:: max(1, len(coords) // 400)]
            world = np.asarray(cache.GetLocalToWorldTransform(prim))
            surfaces.append((np.c_[coords, np.ones(len(coords))] @ world @ back)[:, :3])
        pts = np.concatenate(surfaces) if surfaces else np.empty((0, 3))
        if len(pts):
            pts = pts[pts[:, 2] >= np.quantile(pts[:, 2], 0.35)]
        probes[name] = pts[:: max(1, len(pts) // 1200)]
    return probes


def place(stage, cfg, state, targets, camera):
    from pxr import Gf, Usd, UsdGeom

    now = Usd.TimeCode.Default()

    def set_world(prim, op, world):
        xf = UsdGeom.Xformable(prim)
        parent = (
            Gf.Matrix4d(1)
            if xf.GetResetXformStack()
            else xf.ComputeParentToWorldTransform(now)
        )
        op.Set(world * parent.GetInverse())
        return rows(xf.ComputeLocalToWorldTransform(now))

    cam = state["camera"]
    if "matrix_row" in cam:
        cw = matrix(cam["matrix_row"])
    else:
        cw = (
            Gf.Matrix4d()
            .SetLookAt(
                Gf.Vec3d(*cam["eye_m"]), Gf.Vec3d(*cam["target_m"]), Gf.Vec3d(0, 0, 1)
            )
            .GetInverse()
        )
    effective = {}
    for name, pose in state["objects"].items():
        prim, op, base, corners, _ = targets[name]
        if not pose["visible"]:
            # Visibility toggling can invalidate Replicator semantic IDs when a mesh reappears.
            # Park inactive targets behind the current camera, leaving Hydra registrations intact.
            extent = max(v.GetLength() for v in corners) * max(
                1.0, np.linalg.norm(np.array(base)[:3, :3])
            )
            world = Gf.Matrix4d(base)
            world.SetTranslateOnly(
                cw.Transform(Gf.Vec3d(0, 0, max(50.0, 6.0 * extent)))
            )
            effective[name] = {
                "visible": False,
                "matrix_row": set_world(prim, op, world),
            }
            continue
        if "matrix_row" in pose:
            world = matrix(pose["matrix_row"])
        else:
            rot = Gf.Matrix4d().SetRotate(
                Gf.Rotation(Gf.Vec3d(0, 0, 1), pose["yaw_deg"])
            )
            world = base * rot
            points = np.array([world.Transform(v) for v in corners])
            center = (points.min(axis=0) + points.max(axis=0)) / 2
            center[2] = points[:, 2].min()
            world.SetTranslateOnly(Gf.Vec3d(*(np.array(pose["bottom_xyz_m"]) - center)))
        effective[name] = {"visible": True, "matrix_row": set_world(prim, op, world)}
    cm = set_world(camera.GetPrim(), UsdGeom.Xformable(camera).MakeMatrixXform(), cw)
    w, h = cfg["width"], cfg["height"]
    fx, _, cx, _, fy, cy, _, _, _ = cam["K"]
    focal = 0.35
    ha, va = focal * w / fx, focal * h / fy
    for attr, val in [
        (camera.GetFocalLengthAttr(), focal),
        (camera.GetHorizontalApertureAttr(), ha),
        (camera.GetVerticalApertureAttr(), va),
        (camera.GetHorizontalApertureOffsetAttr(), ha * (0.5 - cx / w)),
        (camera.GetVerticalApertureOffsetAttr(), va * (cy / h - 0.5)),
    ]:
        attr.Set(val)
    f, horiz, vert, ho, vo = [
        float(a.Get())
        for a in (
            camera.GetFocalLengthAttr(),
            camera.GetHorizontalApertureAttr(),
            camera.GetVerticalApertureAttr(),
            camera.GetHorizontalApertureOffsetAttr(),
            camera.GetVerticalApertureOffsetAttr(),
        )
    ]
    if "table_fit" in state:
        mesh = UsdGeom.Mesh.Get(stage, cfg["table_mesh"])
        a, b, c = state["table_fit"]["plane_z_equals_ax_by_c"]
        pts = [
            Gf.Vec3f(x, y, a * x + b * y + c) for x, y, z in mesh.GetPointsAttr().Get()
        ]
        mesh.GetPointsAttr().Set(pts)
        mesh.GetExtentAttr().Set(UsdGeom.PointBased.ComputeExtent(pts))
    return {
        "objects": effective,
        "camera": {
            "matrix_row": cm,
            "K": [
                w * f / horiz,
                0,
                w * (0.5 - ho / horiz),
                0,
                h * f / vert,
                h * (0.5 + vo / vert),
                0,
                0,
                1,
            ],
        },
    }


def property_edits(stage, root, specs, targets=None, parking=None):
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    setters = {}
    files = set()
    for s in specs:
        p = stage.GetPrimAtPath(s["prim"])
        if not p:
            raise ValueError(f"missing intervention prim: {s['prim']}")
        kind = s["kind"]
        if kind in ("scale", "float"):
            attr = p.GetAttribute(s["attribute"])
            if not attr or attr.Get() is None:
                raise ValueError(f"missing {s['attribute']} at {p}")
            base = float(attr.Get()) if kind == "scale" else 1.0

            def apply(value, attr=attr, base=base):
                attr.Set(float(value) * base)
                return float(attr.Get())
        elif kind == "visibility":
            path = str(p.GetPath())
            if any(path == str(target[0].GetPath()) for target in (targets or {}).values()):
                raise ValueError(
                    f"target root visibility is controlled by object pose: {path}"
                )
            descendant = any(
                path.startswith(str(target[0].GetPath()) + "/")
                for target in (targets or {}).values()
            )
            if descendant:
                if not parking or not any(
                    child.IsA(UsdGeom.Xformable) for child in Usd.PrimRange(p)
                ):
                    raise ValueError(
                        f"target visibility edit requires Xformable geometry: {path}"
                    )

                def apply(value, path=path, parking=parking):
                    return parking.set(path, value)
            else:
                attr = UsdGeom.Imageable(p).GetVisibilityAttr()

                def apply(value, attr=attr):
                    attr.Set("inherited" if value else "invisible")
                    return str(attr.Get())
        elif kind in ("variant", "geometry_variant"):
            variants = p.GetVariantSets().GetVariantSet(s["variant_set"])
            if set(s["choices"]) - set(variants.GetVariantNames()):
                raise ValueError(f"unknown variant for {p}")

            def apply(value, variants=variants):
                if not variants.SetVariantSelection(value):
                    raise ValueError(f"unavailable variant {value}")
                return variants.GetVariantSelection()
        elif kind == "binding":
            api = UsdShade.MaterialBindingAPI.Apply(p)
            materials = {v: UsdShade.Material.Get(stage, v) for v in s["choices"]}
            if not all(materials.values()):
                raise ValueError(f"invalid binding for {p}")

            def apply(value, api=api, materials=materials):
                api.Bind(materials[value])
                return str(api.ComputeBoundMaterial()[0].GetPath())
        elif kind == "texture_set":
            # Each named choice supplies a synchronized bundle of albedo/roughness/normal inputs.
            attrs = {k: stage.GetAttributeAtPath(v) for k, v in s["attributes"].items()}
            if not all(attrs.values()):
                raise ValueError(f"missing texture attribute at {p}")
            bundles = {}
            for choice in s["choices"]:
                paths = {
                    k: (root / v).resolve() for k, v in s["bundles"][choice].items()
                }
                if set(paths) != set(attrs) or not all(
                    v.is_file() for v in paths.values()
                ):
                    raise ValueError(f"incomplete texture set {choice}")
                files.update(paths.values())
                bundles[choice] = paths

            def apply(value, bundles=bundles, attrs=attrs):
                for channel, path in bundles[value].items():
                    attrs[channel].Set(Sdf.AssetPath(str(path)))
                    if Path(attrs[channel].Get().path) != path:
                        raise ValueError(f"texture readback failed: {path}")
                return value
        else:
            raise ValueError(f"unknown intervention kind {kind}")
        setters[s["id"]] = apply
    return setters, files


def geometry_sources(stage, configurations):
    """Include every selectable geometry branch, even when it is unloaded at launch."""
    sources = {item["path"]: item for item in source_files(stage)}
    for config in configurations:
        prim = stage.GetPrimAtPath(config["prim"])
        variants = prim.GetVariantSets().GetVariantSet(config["variant_set"])
        initial = variants.GetVariantSelection()
        try:
            for choice in config["choices"]:
                if choice == initial:
                    continue
                if not variants.SetVariantSelection(choice):
                    raise ValueError(f"unknown geometry branch {choice}")
                for item in source_files(stage):
                    sources[item["path"]] = item
        finally:
            if initial:
                variants.SetVariantSelection(initial)
            else:
                variants.ClearVariantSelection()
    return list(sources.values())


def refresh_geometry(stage, targets, configs, selected, probes):
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    readback = {}
    for cfg in configs:
        path = cfg["prim"]
        for name, (prim, op, base, corners, token) in targets.items():
            owner = str(prim.GetPath())
            if path != owner and not path.startswith(owner + "/"):
                continue
            bounds = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
            if bounds.IsEmpty():
                raise ValueError(f"geometry variant has no visible extent: {path}")
            corners = [bounds.GetCorner(i) for i in range(8)]
            targets[name] = (prim, op, base, corners, token)
            probes.update(local_surface_probes({name: targets[name]}))
            readback[name] = {
                "selection": selected[cfg["id"]],
                "local_min": list(bounds.GetMin()),
                "local_max": list(bounds.GetMax()),
            }
    return readback


def synchronized(app, rep, annotators, args):
    # Tight boxes may legitimately be empty for a negative image.
    for attempt in range(16):
        if not app.is_running():
            break
        app.update()
        rep.orchestrator.step(
            rt_subframes=args.rt_subframes,
            pause_timeline=True,
            delta_time=0.0,
            wait_for_render=True,
        )
        data = {k: v.get_data() for k, v in annotators.items()}
        shapes = {}
        for key, plane in data.items():
            a = plane.get("data") if isinstance(plane, dict) else plane
            shapes[key] = np.shape(a)
        if (
            all(
                shapes[k][:2] == (args.height, args.width)
                for k in ("rgb", "semantic", "paths", "depth")
            )
            and isinstance(data["bbox"], dict)
            and "info" in data["bbox"]
        ):
            return data, attempt + 1
    raise RuntimeError(f"native annotators unavailable: {shapes}")


def write_planes(temp, data, header, tokens, effective, probes):
    rgb = np.asarray(data["rgb"])[:, :, :3]
    seg, boxes = data["semantic"], data["bbox"]
    sem = np.asarray(seg["data"]).squeeze()
    depth = np.asarray(data["depth"], dtype=np.float32).squeeze()
    instances = np.zeros(sem.shape, dtype=np.uint16)
    classes = np.zeros(sem.shape, dtype=np.uint8)

    def instance(label):
        labels = set(str(label.get("class", "")).replace(",", " ").split())
        ids = labels.intersection(tokens)
        if len(ids) > 1:
            raise ValueError(f"ambiguous semantic ancestry {ids}")
        return tokens[next(iter(ids))] if ids else 0

    for key, label in seg["info"]["idToLabels"].items():
        ident = instance(label)
        if ident:
            instances[sem == int(key)] = ident
    path_plane = data["paths"]
    path_map = {str(k): str(v) for k, v in path_plane["info"]["idToLabels"].items()}
    geom = np.asarray(path_plane["data"]).squeeze()
    ids_by_path = {spec["prim"]: i for i, spec in enumerate(header["instances"], 1)}
    other_instances = np.zeros_like(instances)
    for k, prim_path in path_map.items():
        for configured, ident in ids_by_path.items():
            if prim_path == configured or prim_path.startswith(configured + "/"):
                other_instances[geom == int(k)] = ident
                break
    disagreement = int(np.sum(other_instances != instances))
    if disagreement:
        raise ValueError(
            f"native class and instance path masks disagree on {disagreement} pixels; paths={path_map}"
        )
    witnessed = {}
    for i, spec in enumerate(header["instances"], 1):
        obj = effective["objects"][spec["id"]]
        observed = bool(np.any(instances == i))
        if observed and not obj["visible"]:
            raise ValueError(f"inactive target in image: {spec['id']}")
        if not observed and obj["visible"]:
            matched = depth_witness(
                probes[spec["id"]], obj["matrix_row"], effective["camera"], depth
            )
            witnessed[spec["id"]] = matched
            if matched >= 12:
                raise ValueError(
                    f"{spec['id']} has {matched} visible projected depth samples but no native mask"
                )
    audit = {
        "instance_path_pixels_different": disagreement,
        "path_ids": path_map,
        "semantic_ids": {str(k): v for k, v in seg["info"]["idToLabels"].items()},
        "unlabelled_object_matches": witnessed,
    }
    native = {}
    for box in boxes["data"]:
        ident = instance(
            boxes["info"]["idToLabels"].get(str(int(box["semanticId"])), {})
        )
        if ident:
            xy = [int(box[k]) for k in ("x_min", "y_min", "x_max", "y_max")]
            native.setdefault(ident, []).append(xy)
    objects = []
    for ident in np.unique(instances):
        if not ident:
            continue
        ident = int(ident)
        spec = header["instances"][ident - 1]
        y, x = np.where(instances == ident)
        category = header["classes"][spec["class"]]
        classes[y, x] = category
        bbox = [
            int(x.min()),
            int(y.min()),
            int(x.max() - x.min() + 1),
            int(y.max() - y.min() + 1),
        ]
        if ident not in native:
            raise ValueError(f"missing RTX tight box for {spec['id']}")
        xy = np.array(native[ident])
        if (
            x.min() < xy[:, 0].min() - 2
            or y.min() < xy[:, 1].min() - 2
            or x.max() > xy[:, 2].max() + 2
            or y.max() > xy[:, 3].max() + 2
        ):
            raise ValueError(
                f"native tight bounds disagree for {spec['id']}: {bbox}, {xy}"
            )
        objects.append(
            {
                "id": spec["id"],
                "instance_id": ident,
                "category_id": category,
                "class": spec["class"],
                "area": len(x),
                "bbox_xywh": bbox,
                "native_tight_xyxy": native[ident],
            }
        )
    output = ("rgb.png", "classes.png", "instances.png", "range_m.npz")
    for name, arr in zip(output, (rgb, classes, instances)):
        Image.fromarray(arr).save(temp / name, compress_level=3)
    np.savez_compressed(temp / output[-1], range_m=depth)
    return objects, {n: digest(temp / n) for n in output}, audit


def run(app, args, root, cfg, cases):
    import carb.settings
    import omni.replicator.core as rep
    import omni.timeline
    from isaacsim.core.experimental.utils.stage import open_stage
    from pxr import UsdGeom

    scene = (root / cfg["stage"]).resolve()
    ok, stage = open_stage(str(scene))
    if not ok:
        raise RuntimeError(f"cannot open {scene}")
    if (
        UsdGeom.GetStageMetersPerUnit(stage) != 1
        or UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z
    ):
        raise ValueError("campaign stage must declare metres and Z-up")
    stage.SetEditTarget(stage.GetSessionLayer())
    targets = bind_targets(stage, root, cfg)
    probes = local_surface_probes(targets)
    camera = UsdGeom.Camera.Get(stage, cfg["camera_prim"])
    if not camera:
        camera = UsdGeom.Camera.Define(stage, cfg["camera_prim"])
    camera.CreateClippingRangeAttr().Set((0.01, 10000))
    camera.GetFStopAttr().Set(0.0)
    configure_camera_exposure(camera)
    parking = DescendantParking(stage, camera)
    setters, extra = property_edits(
        stage, root, cfg.get("edits", []), targets, parking
    )
    geometry_edits = [
        item for item in cfg.get("edits", []) if item["kind"] == "geometry_variant"
    ]
    sources = geometry_sources(stage, geometry_edits)
    # Register initially hidden descendants before the annotators attach.
    for edit in cfg.get("edits", []):
        if edit["kind"] == "visibility" and any(
            edit["prim"].startswith(str(target[0].GetPath()) + "/")
            for target in targets.values()
        ):
            setters[edit["id"]](False)
    extra.add(args.config.resolve())
    if "poses" in cfg:
        extra.add((root / cfg["poses"]).resolve())
    for path in extra:
        sources.append(
            {"path": os.path.relpath(path, scene.parent), "sha256": digest(path)}
        )
    app.reset_render_settings()
    settings = carb.settings.get_settings()
    render = configure_render(settings, args)
    header = {
        "format": 2,
        "seed": args.seed,
        "classes": cfg["classes"],
        "instances": cfg["instances"],
        "texture_sets": texture_registry(root, cfg.get("edits", [])),
        "sources": sorted(sources, key=lambda v: v["path"]),
        "renderer": render,
        "resolution": [args.width, args.height],
        "rt_subframes": args.rt_subframes,
        "warmup_frames": args.warmup_frames,
        "engine": renderer_versions(),
        "code": {
            p.name: digest(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("capture_isaac.py"),
                Path(__file__).with_name("generate_dataset.py"),
                Path(__file__).resolve().parents[1] / "harness/campaign.py",
                Path(__file__).resolve().parents[1] / "harness/visibility.py",
            )
        },
        "depth": "Euclidean metres from optical center; background may be infinite",
        "exposure": {
            "iso": 100,
            "time": 1,
            "fstop": 5,
            "responsivity": 1,
            "compensation_ev": 0,
        },
    }
    header["campaign_key"] = hashlib.sha256(
        json.dumps(header, sort_keys=True).encode()
    ).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    set_header(args.out, header)
    frames = args.out / "frames"
    frames.mkdir(exist_ok=True)
    rep.set_global_seed(args.seed)
    rep.orchestrator.set_capture_on_play(False)
    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()
    timeline.commit()
    product = rep.create.render_product(cfg["camera_prim"], (args.width, args.height))
    params = {"semanticTypes": ["class"]}
    annotators = {
        "rgb": rep.annotators.get("rgb"),
        "semantic": rep.annotators.get(
            "semantic_segmentation", init_params={**params, "colorize": False}
        ),
        "paths": rep.annotators.get(
            "instance_segmentation_fast", init_params={**params, "colorize": False}
        ),
        "bbox": rep.annotators.get("bounding_box_2d_tight", init_params=params),
        "depth": rep.annotators.get("distance_to_camera"),
    }
    for a in annotators.values():
        a.attach(product)
    tokens = {v[-1]: i for i, v in enumerate(targets.values(), 1)}
    try:
        for _ in range(args.warmup_frames):
            app.update()
        for index in range(args.start, args.start + args.count):
            frame = frames / f"{index:08d}"
            expected = sample(cfg, cases, args.seed, index)
            with lock(args.out / ".locks" / frame.name):
                if frame.exists():
                    old = inspect_frame(frame, header)
                    if old["sample"] != expected:
                        raise ValueError(f"{frame}: sample mismatch")
                    print(f"verified existing frame {index}", flush=True)
                    continue
                parking.restore()
                geometry_values = {
                    s["id"]: setters[s["id"]](expected["edits"][s["id"]])
                    for s in geometry_edits
                }
                geometry = (
                    refresh_geometry(
                        stage, targets, geometry_edits, geometry_values, probes
                    )
                    if geometry_edits
                    else {}
                )
                effective = place(stage, cfg, expected, targets, camera)
                effective["edits"] = {
                    **geometry_values,
                    **{
                        k: setters[k](v)
                        for k, v in expected["edits"].items()
                        if k not in geometry_values
                    },
                }
                if geometry:
                    effective["geometry"] = geometry
                settings.set("/rtx-transient/post/dlss/forceParamReset", True)
                payloads, steps = synchronized(app, rep, annotators, args)
                with TemporaryDirectory(prefix=".pending-", dir=frames) as work:
                    temp = Path(work) / frame.name
                    temp.mkdir()
                    objects, hashes, native_audit = write_planes(
                        temp, payloads, header, tokens, effective, probes
                    )
                    write(
                        temp / "annotation.json",
                        {
                            "campaign_key": header["campaign_key"],
                            "index": index,
                            "sample": expected,
                            "readback": effective,
                            "objects": objects,
                            "sha256": hashes,
                            "native_audit": native_audit,
                            "annotator_steps": steps,
                        },
                    )
                    (temp / "record.sha256").write_text(
                        digest(temp / "annotation.json") + "\n"
                    )
                    inspect_frame(temp, header)
                    temp.rename(frame)
                print(
                    f"captured frame {index} with {len(objects)} visible objects",
                    flush=True,
                )
    finally:
        for a in annotators.values():
            a.detach()
        product.destroy()
        rep.orchestrator.wait_until_complete()


if __name__ == "__main__":
    main()
