"""Author a relocatable USD variant family from existing metered, Z-up assets."""

import argparse
import os
import re
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom


def _asset_dependencies(stage, path):
    """Reject unavailable composition and authored asset dependencies."""
    for layer in stage.GetUsedLayers():
        if layer.anonymous:
            continue
        for dependency in (*layer.GetExternalReferences(), *layer.GetExternalAssetDependencies()):
            if not dependency:
                continue
            resolved = layer.ComputeAbsolutePath(dependency)
            if not Path(resolved).exists():
                raise ValueError(f"missing dependency in {path}: {dependency}")
    for prim in stage.Traverse():
        for attribute in prim.GetAttributes():
            if attribute.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attribute.Get()
            if value and value.path and not value.resolvedPath:
                raise ValueError(f"unresolved asset in {path}: {prim.GetPath()}.{attribute.GetName()} = {value.path}")


def validate_asset(path):
    path = Path(path).resolve(strict=True)
    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise ValueError(f"cannot open USD: {path}")
    if stage.GetCompositionErrors():
        raise ValueError(f"unresolved USD composition in {path}: {stage.GetCompositionErrors()}")
    if UsdGeom.GetStageUpAxis(stage) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError(f"asset must declare Z-up and metersPerUnit = 1: {path}")
    prim = stage.GetDefaultPrim()
    if not prim or not prim.IsA(UsdGeom.Xformable):
        raise ValueError(f"asset needs a transformable default prim: {path}")
    _asset_dependencies(stage, path)
    extent = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    ).ComputeWorldBound(prim).ComputeAlignedRange()
    if extent.IsEmpty():
        raise ValueError(f"asset default prim has no visible extent: {path}")
    return path, float(extent.GetMin()[2])


def create_family(variants, out, variant_set="geometry", default=None):
    if not variants:
        raise ValueError("supply at least one variant")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", variant_set):
        raise ValueError("variant set must be a USD identifier")
    names = [name for name, _ in variants]
    if len(names) != len(set(names)) or any(
        not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in names
    ):
        raise ValueError("variant names must be distinct USD identifiers")
    if default is None:
        default = names[0]
    if default not in names:
        raise ValueError(f"default variant is not supplied: {default}")
    assets = [(name, *validate_asset(path)) for name, path in variants]
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    choices = root.GetPrim().GetVariantSets().AddVariantSet(variant_set)
    for name, source, bottom in assets:
        choices.AddVariant(name)
        choices.SetVariantSelection(name)
        with choices.GetVariantEditContext():
            model = UsdGeom.Xform.Define(stage, "/Asset/Geometry")
            model.GetPrim().GetReferences().AddReference(os.path.relpath(source, out.parent))
            model.AddTranslateOp().Set(Gf.Vec3d(0, 0, -bottom))
    choices.SetVariantSelection(default)
    stage.GetRootLayer().Save()
    result = Usd.Stage.Open(str(out))
    if not result or not result.GetDefaultPrim() or result.GetDefaultPrim().GetVariantSets().GetVariantSet(variant_set).GetVariantSelection() != default:
        raise ValueError(f"invalid authored family: {out}")
    return out


def _variant_arg(value):
    name, sep, path = value.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError("use NAME=PATH")
    return name, Path(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", type=_variant_arg, required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--variant-set", default="geometry")
    parser.add_argument("--default")
    args = parser.parse_args(argv)
    print(create_family(args.variant, args.out, args.variant_set, args.default))


if __name__ == "__main__":
    main()
