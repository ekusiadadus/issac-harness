from pathlib import Path
import math

import numpy as np
import pytest
from PIL import Image
from pxr import Gf, Usd, UsdGeom

from harness import campaign
from harness.surface_mark import add_mark, alpha_rectangles, main


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/inspection/scene.usda"
TEXTURE = ROOT / "examples/campaign/handling-mark.png"


def test_mark_geometry_and_rejections(tmp_path):
    stage = Usd.Stage.Open(str(SOURCE))
    stage.SetEditTarget(stage.GetSessionLayer())
    path = "/World/Props/Carton"
    mesh = path + "/WallFront"
    mark = add_mark(stage, mesh, 31, path, str(TEXTURE), 0.060, 0.037)
    assert len(mark.GetPointsAttr().Get()) > 4
    assert mark.GetPrim().GetPath().pathString == mesh + "/HandlingMark"
    assert UsdGeom.PrimvarsAPI(mark).GetPrimvar("st").Get() is not None
    with pytest.raises(ValueError, match="already exists"):
        add_mark(stage, mesh, 31, path, str(TEXTURE), 0.060, 0.037)
    with pytest.raises(ValueError, match="boundary"):
        add_mark(stage, mesh, 31, path, str(TEXTURE), 0.5, 0.037, name="TooBig")
    with pytest.raises(ValueError, match="quad"):
        add_mark(stage, mesh, 10000, path, str(TEXTURE), 0.01, 0.01, name="BadFace")


def test_transformed_scaled_mesh_and_left_handed(tmp_path):
    opaque = tmp_path / "opaque.png"
    Image.new("RGBA", (4, 4), (220, 10, 10, 255)).save(opaque)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    parent = UsdGeom.Xform.Define(stage, "/Owner")
    parent.AddTranslateOp().Set(Gf.Vec3d(2, 3, 4))
    parent.AddScaleOp().Set(Gf.Vec3f(2, 3, 1))
    nested = UsdGeom.Xform.Define(stage, "/Owner/Nested")
    nested.AddRotateZOp().Set(30)
    mesh = UsdGeom.Mesh.Define(stage, "/Owner/Nested/Face")
    mesh.CreatePointsAttr([(-10, -10, 0), (10, -10, 0), (10, 10, 0), (-10, 10, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateOrientationAttr(UsdGeom.Tokens.leftHanded)
    mark = add_mark(stage, str(mesh.GetPath()), 0, "/Owner", str(opaque), 0.06, 0.04)
    assert str(mark.GetPath()) == "/Owner/Nested/Face/HandlingMark"
    xform = UsdGeom.XformCache().GetLocalToWorldTransform(mark.GetPrim())
    world = [np.array(xform.Transform(Gf.Vec3d(*p))) * 0.01 for p in mark.GetPointsAttr().Get()]
    assert np.linalg.norm(world[1] - world[0]) == pytest.approx(0.06, abs=1e-6)
    assert np.linalg.norm(world[3] - world[0]) == pytest.approx(0.04, abs=1e-6)
    assert world[0][2] == pytest.approx(0.04 - 0.0008, abs=1e-6)
    with pytest.raises(ValueError, match="finite"):
        add_mark(stage, str(mesh.GetPath()), 0, "/Owner", str(opaque), math.nan, 0.04, name="Bad")
    mesh.GetPointsAttr().Set([(0, 0, 0), (1, 0, 0), (0.25, 0.25, 0), (0, 1, 0)])
    with pytest.raises(ValueError, match="convex"):
        add_mark(stage, str(mesh.GetPath()), 0, "/Owner", str(opaque), 0.01, 0.01, name="Concave")


def test_alpha_cutout_preserves_interior_void_and_uv(tmp_path):
    alpha = np.full((5, 5, 4), 255, dtype=np.uint8)
    alpha[2, 2, 3] = 0
    alpha[0, 0, 3] = 127
    texture = tmp_path / "ring.png"
    Image.fromarray(alpha, "RGBA").save(texture)
    rectangles = alpha_rectangles(Image.open(texture))
    assert len(rectangles) < 24  # vertical span compression
    assert sum((u1 - u0) * (v1 - v0) for u0, v0, u1, v1 in rectangles) == pytest.approx(23 / 25)
    assert not any(u0 < 0.5 < u1 and v0 < 0.5 < v1 for u0, v0, u1, v1 in rectangles)
    stage = Usd.Stage.Open(str(SOURCE))
    stage.SetEditTarget(stage.GetSessionLayer())
    mark = add_mark(stage, "/World/Props/Carton/WallFront", 31,
                    "/World/Props/Carton", str(texture), 0.06, 0.037)
    assert len(mark.GetFaceVertexCountsAttr().Get()) == len(rectangles)
    uv = UsdGeom.PrimvarsAPI(mark).GetPrimvar("st").Get()
    assert len(uv) == 4 * len(rectangles)
    surface = stage.GetPrimAtPath("/World/Props/Carton/HandlingMarkMaterial/PreviewSurface")
    assert not surface.GetAttribute("inputs:opacity")
    assert not surface.GetAttribute("inputs:opacityThreshold")


def test_example_cpu_composition():
    config = ROOT / "examples/campaign/local-mark/production.toml"
    root, cfg = campaign.load(config)
    stage = Usd.Stage.Open(str(root / cfg["stage"]))
    assert stage.GetPrimAtPath("/World/Props/Carton/WallFront/HandlingMark").IsValid()
    mark = UsdGeom.Mesh.Get(stage, "/World/Props/Carton/WallFront/HandlingMark")
    assert len(mark.GetFaceVertexCountsAttr().Get()) > 1
    assert not stage.GetPrimAtPath(
        "/World/Props/Carton/HandlingMarkMaterial/PreviewSurface"
    ).GetAttribute("inputs:opacity")
    assert stage.GetDefaultPrim().GetPath().pathString == "/World"
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert cfg["instances"][0]["prim"] == "/World/Props/Carton"
    assert all((root / v).is_file() for e in cfg["edits"] if e["id"] == "paper"
               for bundle in e["bundles"].values() for v in bundle.values())


def test_label_occlusion_rejected_and_fixed_pair():
    stage = Usd.Stage.Open(str(SOURCE))
    stage.SetEditTarget(stage.GetSessionLayer())
    args = (stage, "/World/Props/Carton/WallFront", 31,
            "/World/Props/Carton", str(TEXTURE), 0.060, 0.037)
    label = ("/World/Props/Carton/LabelPrint",)
    with pytest.raises(ValueError, match="overlaps keep-clear"):
        add_mark(*args, center_uv=(0.5, 0.57), keep_clear=label)
    add_mark(*args, center_uv=(0.17, 0.40), keep_clear=label)
    config = ROOT / "examples/campaign/local-mark/compare.toml"
    root, cfg = campaign.load(config)
    rows = campaign.read(root / cfg["poses"])
    assert len(rows) == 3
    assert rows[0] == rows[2]
    assert rows[0]["camera"] == rows[1]["camera"]
    assert rows[0]["objects"] == rows[1]["objects"]
    assert rows[0]["edits"] | {"mark_visible": True} == rows[1]["edits"]
    assert rows[0]["edits"]["mark_visible"] is False
    assert rows[0]["objects"]["crate"] == {
        "bottom_xyz_m": [0.0, 0.0, 0.8925], "yaw_deg": 0.0, "visible": True}


def test_cli_writes_new_portable_layer(tmp_path):
    output = tmp_path / "marked.usda"
    main([str(SOURCE), str(output), "--mesh", "/World/Props/Carton/WallFront",
          "--face", "31", "--owner", "/World/Props/Carton",
          "--texture", str(ROOT / "examples/campaign/handling-mark.png"),
          "--width-mm", "60", "--height-mm", "37"])
    stage = Usd.Stage.Open(str(output))
    assert stage.GetPrimAtPath("/World/Props/Carton/WallFront/HandlingMark")
    assert stage.GetAttributeAtPath(
        "/World/Props/Carton/HandlingMarkMaterial/Texture.inputs:file"
    ).Get().resolvedPath
    with pytest.raises(SystemExit):
        main([str(SOURCE), str(output), "--mesh", "/World/Props/Carton/WallFront",
              "--face", "31", "--owner", "/World/Props/Carton",
              "--texture", str(ROOT / "examples/campaign/handling-mark.png"),
              "--width-mm", "60", "--height-mm", "37"])
