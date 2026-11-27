"""Regressions that a successful USD open or thumbnail alone would miss."""

import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from harness.audit import audit
from harness.project import load_project
from harness.render import inspect_frame

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/inspection"


class SceneChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name).resolve() / "a relocated scene"
        shutil.copytree(EXAMPLE, cls.root)
        cls.project = load_project(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def override(self):
        path = self.root / (self._testMethodName + ".usda")
        stage = Usd.Stage.CreateNew(str(path))
        stage.GetRootLayer().subLayerPaths = ["scene.usda"]
        stage.SetMetadata("defaultPrim", "World")
        UsdGeom.SetStageUpAxis(stage, "Z")
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        return stage, replace(self.project, cases={"test": path})

    def test_relocated_pair_changes_only_overhead_exposure(self):
        report = audit(self.project)
        self.assertEqual(report["status"], "pass", report["cases"])
        base, dim = (report["cases"][name] for name in ("nominal", "low_overhead"))
        self.assertEqual(base["views"], dim["views"])
        self.assertEqual(base["materials"], dim["materials"])
        for light, settings in base["lights"].items():
            expected = dict(settings)
            if light == "/World/Lights/OverheadSoftbox":
                expected["exposure"] -= 2
            self.assertEqual(expected, dim["lights"][light])
        self.assertGreater(base["views"]["detail"]["projected_width_px"], 250)
        self.assertEqual(len(base["supports"]), 4)

    def test_connected_maps_units_and_support(self):
        stage, project = self.override()
        prim = stage.GetPrimAtPath("/World/Props/Carton/Label")
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        surface, _, _ = material.ComputeSurfaceSource()
        source, _, _ = surface.GetInput("normal").GetConnectedSource()
        texture = UsdShade.Shader(source.GetPrim())
        texture.GetInput("sourceColorSpace").Set("sRGB")
        texture.GetInput("file").Set(Sdf.AssetPath("./lost_normal.png"))
        cam = UsdGeom.Camera.Get(stage, project.views["detail"].camera)
        # Same field of view, but accidentally entered millimeters as raw USD camera units.
        for attr in (
            cam.GetFocalLengthAttr(),
            cam.GetHorizontalApertureAttr(),
            cam.GetVerticalApertureAttr(),
        ):
            attr.Set(100 * attr.Get())
        UsdGeom.Xformable(stage.GetPrimAtPath("/World/Props/Handset")).AddTranslateOp(opSuffix="test").Set(
            Gf.Vec3d(0, 0, 0.012)
        )
        stage.GetRootLayer().Save()
        report = audit(project)["cases"]["test"]
        errors = "\n".join(report["errors"])
        self.assertEqual(report["status"], "fail")
        for message in ("lost_normal.png", "explicit raw", "mm lens", "vertical gap"):
            self.assertIn(message, errors)

    def test_inherited_material_and_hidden_proxy(self):
        stage, project = self.override()
        mesh = stage.GetPrimAtPath("/World/Props/Carton/Label")
        material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
        UsdShade.MaterialBindingAPI.Apply(mesh.GetParent()).Bind(material)
        UsdShade.MaterialBindingAPI(mesh).UnbindDirectBinding()
        proxy = UsdGeom.Cube.Define(stage, "/World/OnlyCollision")
        proxy.CreatePurposeAttr("proxy")
        stage.GetRootLayer().Save()
        self.assertEqual(audit(project)["status"], "pass")
        proxy.GetPurposeAttr().Set("render")
        stage.GetRootLayer().Save()
        errors = audit(project)["cases"]["test"]["errors"]
        self.assertTrue(any("OnlyCollision" in e and "no bound material" in e for e in errors), errors)

    def test_rgb_constant_frame_does_not_pass(self):
        path = self.root / "red.png"
        Image.new("RGB", (32, 32), (235, 20, 28)).save(path)
        with self.assertRaisesRegex(ValueError, "constant RGB"):
            inspect_frame(path, [32, 32])

    def test_collapsed_uvs_and_cropped_feet(self):
        stage, project = self.override()
        mesh = stage.GetPrimAtPath("/World/Environment/Worktop")
        uv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
        uv.Set([Gf.Vec2f(i, i) for i in range(len(uv.Get()))])
        camera = UsdGeom.Camera.Get(stage, project.views["overview"].camera)
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(1.68, -2.15, 1.62), Gf.Vec3d(0, 0, 0.82), Gf.Vec3d(0, 0, 1))
        camera.GetPrim().GetAttribute("xformOp:transform").Set(view.GetInverse())
        stage.GetRootLayer().Save()
        errors = "\n".join(audit(project)["cases"]["test"]["errors"])
        self.assertIn("UVs collapse", errors)
        self.assertIn("is cropped", errors)

    def test_barcode_must_be_visible_in_the_target_region(self):
        path = self.root / "two-regions.png"
        sheet = Image.new("RGB", (2600, 590), "white")
        with Image.open(self.root / "textures/label_print_base.png") as label:
            sheet.paste(label, (0, 0))
        sheet.save(path)
        opts = {"barcode": self.project.views["detail"].expected_barcode}
        left = inspect_frame(path, [2600, 590], target_rect=[0, 0, 0.5, 1], **opts)
        right = inspect_frame(path, [2600, 590], target_rect=[0.5, 0, 1, 1], **opts)
        self.assertTrue(left["barcode"]["matches"])
        self.assertFalse(right["barcode"]["matches"])
