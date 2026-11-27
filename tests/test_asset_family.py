from pathlib import Path

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom

from harness.asset_family import create_family
from harness.campaign import load, sample


def _asset(path, height=0.2, texture=None):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    root = UsdGeom.Xform.Define(stage, "/Item")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Item/Box")
    mesh.CreatePointsAttr([Gf.Vec3f(-0.02, -0.02, -height / 2), Gf.Vec3f(0.02, 0.02, height / 2)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 1])
    if texture:
        root.GetPrim().CreateAttribute("asset:photo", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
    stage.GetRootLayer().Save()


def test_family_switches_geometry_and_keeps_relative_references(tmp_path):
    source = tmp_path / "models"
    source.mkdir()
    _asset(source / "a.usda", height=0.2)
    _asset(source / "b.usda", height=0.1)
    out = tmp_path / "family" / "package.usda"
    create_family([("tall", source / "a.usda"), ("short", source / "b.usda")], out)
    authored = out.read_text()
    assert "../models/a.usda" in authored and "../models/b.usda" in authored
    assert str(tmp_path) not in authored
    stage = Usd.Stage.Open(str(out))
    prim = stage.GetDefaultPrim()
    variant = prim.GetVariantSets().GetVariantSet("geometry")
    assert variant.GetVariantSelection() == "tall"
    for name, height in (("tall", 0.2), ("short", 0.1)):
        variant.SetVariantSelection(name)
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bound = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        assert bound.GetMin()[2] == pytest.approx(0)
        assert bound.GetMax()[2] == pytest.approx(height)
    assert not stage.GetCompositionErrors()


def test_family_rejects_missing_texture_and_duplicate_names(tmp_path):
    source = tmp_path / "item.usda"
    _asset(source, texture="lost.jpg")
    with pytest.raises(ValueError, match="unresolved asset"):
        create_family([("one", source)], tmp_path / "family.usda")
    (tmp_path / "lost.jpg").write_bytes(b"image")
    with pytest.raises(ValueError, match="distinct"):
        create_family([("one", source), ("one", source)], tmp_path / "family.usda")


def test_example_has_two_independent_variant_instances():
    root = Path(__file__).resolve().parents[1] / "examples/campaign/families"
    stage = Usd.Stage.Open(str(root / "scene.usda"))
    assert stage and not stage.GetCompositionErrors()
    for name in ("A", "B"):
        prim = stage.GetPrimAtPath(f"/World/Packages/{name}")
        variant = prim.GetVariantSets().GetVariantSet("geometry")
        assert set(variant.GetVariantNames()) == {"cracker", "sugar"}
        for choice in variant.GetVariantNames():
            variant.SetVariantSelection(choice)
            bounds = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
            ).ComputeWorldBound(prim).ComputeAlignedRange()
            assert bounds.GetMin()[2] == pytest.approx(0)
            assert not bounds.IsEmpty()
    _, config = load(root / "capture/campaign.toml")
    values = [sample(config, None, 0, index)["edits"] for index in range(8)]
    for key in ("package_a_geometry", "package_b_geometry"):
        assert {row[key] for row in values} == {"cracker", "sugar"}
