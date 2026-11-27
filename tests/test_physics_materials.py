"""Reusable prop boundaries and physical contact checks."""

import unittest
import random
import tempfile
from pathlib import Path

from PIL import Image
from pxr import Usd, UsdPhysics, UsdShade

from harness.audit import audit
from harness.materials import image_info
from harness.project import load_project


ROOT = Path(__file__).resolve().parents[1] / "examples" / "inspection"


class PhysicsMaterialsChecks(unittest.TestCase):
    def test_extreme_pixel_does_not_make_weak_noise_meaningful(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "weak.png"
            rng = random.Random(7)
            image = Image.new("RGB", (128, 128))
            image.putdata([
                (max(0, min(255, round(128 + rng.gauss(0, 2)))), 128, 255)
                for _ in range(128 * 128)
            ])
            image.putpixel((0, 0), (0, 255, 0))
            image.save(path)
            info = image_info(path, "rgb", root, {}, "rg")
            self.assertGreater(info["range"], 200)
            self.assertLess(info["p95_p05_range"], 12)

    def test_component_assets_are_self_contained(self):
        for name in ("carton", "handset", "tray"):
            with self.subTest(name=name):
                stage = Usd.Stage.Open(str(ROOT / "assets" / name / f"{name}.usdc"))
                self.assertFalse(stage.GetCompositionErrors())
                body = stage.GetDefaultPrim()
                self.assertEqual(body.GetName().lower(), name)
                self.assertEqual(Usd.ModelAPI(body).GetKind(), "component")
                self.assertTrue(body.HasAPI(UsdPhysics.RigidBodyAPI))
                self.assertGreater(UsdPhysics.MassAPI(body).GetMassAttr().Get(), 0)
                self.assertIn("SemanticsLabelsAPI:class", body.GetAppliedSchemas())
                self.assertTrue(body.GetAttribute("semantics:labels:class").Get())
                self.assertEqual(tuple(body.GetAttribute("xformOp:translate").Get()), (0, 0, 0))
                colliders = [p for p in Usd.PrimRange(body) if p.HasAPI(UsdPhysics.CollisionAPI)]
                self.assertTrue(colliders)
                for prim in Usd.PrimRange(body):
                    if prim.IsA(UsdShade.Shader):
                        file_input = UsdShade.Shader(prim).GetInput("file")
                        if file_input:
                            file = Path(file_input.Get().path)
                            self.assertTrue((ROOT / "assets" / name / file).is_file())
                meta = stage.GetRootLayer().customLayerData["SimReady_Metadata"]
                self.assertEqual(meta["asset_license"], "NOASSERTION")
                self.assertTrue(str(meta["qcode"]).startswith("Q"))

    def test_surface_variant_changes_texture_response(self):
        stage = Usd.Stage.Open(str(ROOT / "scene.usda"))
        body = stage.GetPrimAtPath("/World/Props/Carton")
        finish = body.GetVariantSets().GetVariantSet("surface_finish")
        self.assertEqual(finish.GetVariantSelection(), "standard")
        shader = UsdShade.Shader.Get(stage, "/World/Props/Carton/Materials/Carton/Roughness")
        original = shader.GetInput("bias").Get()
        finish.SetVariantSelection("worn")
        self.assertNotEqual(shader.GetInput("bias").Get(), original)

    def test_scene_rigid_bodies_and_side_walls(self):
        report = audit(load_project(ROOT))
        self.assertEqual(report["status"], "pass", report["cases"])
        bodies = report["cases"]["nominal"]["rigid_bodies"]
        for name in ("Carton", "Handset", "Tray"):
            with self.subTest(name=name):
                self.assertIn(f"/World/Props/{name}", bodies)
        self.assertEqual(len(bodies["/World/Props/Carton"]["colliders"]), 5)
        self.assertEqual(len(bodies["/World/Props/Tray"]["colliders"]), 5)


if __name__ == "__main__":
    unittest.main()
