"""Create a campaign for an existing metered, Z-up USD scene."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

from .campaign import load


def target_argument(value):
    label, equal, path = value.partition("=")
    if (
        not equal
        or not label.strip()
        or not label.isprintable()
        or not Sdf.Path.IsValidPathString(path)
        or not Sdf.Path(path).IsAbsoluteRootOrPrimPath()
        or path == "/"
    ):
        raise argparse.ArgumentTypeError("use class_name=/absolute/UsdPrim")
    return label.strip(), path


def labelled_targets(source):
    stage = Usd.Stage.Open(str(Path(source).resolve(strict=True)))
    result = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Xformable):
            continue
        attribute = prim.GetAttribute("semantics:labels:class")
        names = attribute.Get() if attribute else None
        if not names:
            continue
        if len(names) != 1:
            raise ValueError(
                f"multiple class labels at {prim.GetPath()}; select targets explicitly"
            )
        name, path = str(names[0]), str(prim.GetPath())
        parent = next((c for c, p in result if path.startswith(p + "/")), None)
        if parent == name:
            continue
        if parent is not None:
            raise ValueError(
                f"ambiguous nested class {name} under {parent} at {path}; select targets explicitly"
            )
        result.append((name, path))
    if not result:
        raise ValueError("no authored SemanticsLabelsAPI:class targets found")
    return result


def studio(path, source, stage, target, span):
    layer = Usd.Stage.CreateNew(str(path))
    layer.GetRootLayer().subLayerPaths.append(os.path.relpath(source, path.parent))
    UsdGeom.SetStageUpAxis(layer, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(layer, 1)
    if layer.GetPrimAtPath("/_HarnessStudio"):
        raise ValueError("studio path /_HarnessStudio already exists")
    for key, vector, intensity, size in (
        ("Key", (-0.8, -1.1, 1.8), 475, 0.85),
        ("Fill", (1.1, 0.3, 1.25), 212.5, 1.30),
    ):
        light = UsdLux.RectLight.Define(layer, "/_HarnessStudio/" + key)
        xyz = target + Gf.Vec3d(*vector) * span
        light.AddTransformOp().Set(
            Gf.Matrix4d().SetLookAt(xyz, target, Gf.Vec3d(0, 0, 1)).GetInverse()
        )
        light.CreateWidthAttr(size * span)
        light.CreateHeightAttr(size * span * 0.7)
        light.CreateIntensityAttr(intensity)
    default = stage.GetDefaultPrim()
    if default:
        layer.SetDefaultPrim(layer.GetPrimAtPath(default.GetPath()))
    layer.GetRootLayer().Save()


def initialize(source, out, targets, width=1280, height=960, force_studio=False):
    source = source.resolve(strict=True)
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f"choose a new project directory: {out}")
    if min(width, height) < 1:
        raise ValueError("resolution must be positive")
    stage = Usd.Stage.Open(str(source))
    if (
        not stage
        or UsdGeom.GetStageUpAxis(stage) != "Z"
        or UsdGeom.GetStageMetersPerUnit(stage) != 1
    ):
        raise ValueError("author a Z-up stage with metersPerUnit = 1")
    if stage.GetPrimAtPath("/_HarnessCapture"):
        raise ValueError("source already uses /_HarnessCapture")
    prim_names = [p for _, p in targets]
    if len(set(prim_names)) != len(prim_names) or any(
        p.startswith(q + "/") for p in prim_names for q in prim_names
    ):
        raise ValueError("each capture target must be a distinct, non-nested prim")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    obj = []
    bounds = Gf.Range3d()
    for index, (category, prim_path) in enumerate(targets, 1):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsA(UsdGeom.Xformable):
            raise ValueError(f"target cannot be transformed: {prim_path}")
        extent = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if extent.IsEmpty():
            raise ValueError(f"missing visible extent: {prim_path}")
        point = extent.GetMidpoint()
        point[2] = extent.GetMin()[2]
        bounds.UnionWith(extent)
        obj.append((f"object{index}", category, prim_path, [float(v) for v in point]))
    if bounds.IsEmpty():
        raise ValueError("at least one target is required")
    aim = bounds.GetMidpoint()
    span = max(0.03, bounds.GetSize().GetLength())
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".author-", dir=out.parent) as working:
        pack = Path(working) / out.name
        pack.mkdir()
        use_studio = force_studio or not any(
            p.HasAPI(UsdLux.LightAPI) for p in stage.Traverse()
        )
        stage_name = os.path.relpath(source, out)
        if use_studio:
            stage_name = "studio.usda"
            # Ship a stable relative sublayer even though authoring is staged in a temporary directory.
            studio(pack / stage_name, source, stage, aim, span)
            lay = Sdf.Layer.FindOrOpen(str(pack / stage_name))
            lay.subLayerPaths = [os.path.relpath(source, out)]
            lay.Save()

        def table(name):
            return ["", f"[[{name}]]"]

        lines = [
            "# Starter camera distances, lighting and jitter are authored estimates. Set ranges from your deployment.",
            f"stage = {json.dumps(stage_name)}",
            'camera_prim = "/_HarnessCapture/Camera"',
            f"width = {width}",
            f"height = {height}",
            "",
            "[classes]",
        ]
        for i, cat in enumerate(dict.fromkeys(o[1] for o in obj), 1):
            lines.append(f"{json.dumps(cat)} = {i}")
        for ident, cat, prim, _ in obj:
            lines += table("instances") + [
                f'id = "{ident}"',
                f"class = {json.dumps(cat)}",
                f"prim = {json.dumps(prim)}",
                "jitter_xy_m = [0.0, 0.0]",
                "yaw_deg = [0.0, 0.0]",
            ]
        lines += table("layouts") + ['id = "authored"', "[layouts.anchors]"]
        lines += [f"{ident} = {center}" for ident, _, _, center in obj]
        focal = width / 0.90
        for name, az, el in [
            ("oblique", [-125, -63], [22, 48]),
            ("top", [-115, -67], [62, 82]),
        ]:
            lines += table("cameras") + [
                f'id = "{name}"',
                f"target = {list(aim)}",
                f"azimuth_deg = {az}",
                f"elevation_deg = {el}",
                f"distance_m = {[span * 1.7, span * 2.3]}",
                f"focal_px = {[0.94 * focal, 1.06 * focal]}",
            ]
        if use_studio:
            lines += table("edits") + [
                'id = "key_intensity"',
                'kind = "scale"',
                'prim = "/_HarnessStudio/Key"',
                'attribute = "inputs:intensity"',
                "uniform = [0.8, 1.2]",
            ]
        (pack / "campaign.toml").write_text("\n".join(lines) + "\n")
        pack.rename(out)
    load(out / "campaign.toml")
    return out / "campaign.toml"


def main(argv):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("scene", type=Path)
    cli.add_argument("--out", required=True, type=Path)
    targets = cli.add_mutually_exclusive_group(required=True)
    targets.add_argument("--target", action="append", type=target_argument)
    targets.add_argument(
        "--from-semantics",
        action="store_true",
        help="Register class-labelled prims in traversal order",
    )
    cli.add_argument("--width", type=int, default=1280)
    cli.add_argument("--height", type=int, default=960)
    cli.add_argument(
        "--studio-lighting",
        action="store_true",
        help="Add a three-quarter key and fill even if authored lights exist",
    )
    a = cli.parse_args(argv)
    selected = labelled_targets(a.scene) if a.from_semantics else a.target
    print(initialize(a.scene, a.out, selected, a.width, a.height, a.studio_lighting))
