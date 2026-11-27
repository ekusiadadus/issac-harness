import unittest
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom

from harness.visibility import DescendantParking

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from runtime.capture_campaign import property_edits


def matrix(prim):
    return np.asarray(
        UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    )


class VisibilityPlacement(unittest.TestCase):
    def scene(self, reset=False, initially_visible=False):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        parent = UsdGeom.Xform.Define(stage, "/World")
        parent.AddTranslateOp().Set((2, 3, 0))
        parent.AddRotateZOp().Set(35)
        parent.AddScaleOp().Set((2, 0.5, 1.5))
        target = UsdGeom.Xform.Define(stage, "/World/Target")
        target.AddTranslateOp().Set((0.4, 0, 0))
        child = UsdGeom.Cube.Define(stage, "/World/Target/Mark")
        child.AddTranslateOp().Set((0.2, 0.1, 0.3))
        child.AddRotateXOp().Set(17)
        if reset:
            child.SetResetXformStack(True)
        child.CreateVisibilityAttr().Set(
            "inherited" if initially_visible else "invisible"
        )
        camera = UsdGeom.Camera.Define(stage, "/Camera")
        camera.AddTranslateOp().Set((5, -1, 2))
        camera.AddRotateYOp().Set(15)
        stage.SetEditTarget(stage.GetSessionLayer())
        return stage, target, child, camera

    def test_hidden_visible_repeatability_and_exact_authored_ops(self):
        for reset in (False, True):
            for initially_visible in (False, True):
                self.check_round_trip(reset, initially_visible)

    def check_round_trip(self, reset, initially_visible):
        with self.subTest(reset=reset, initially_visible=initially_visible):
            stage, target, child, camera = self.scene(reset, initially_visible)
            original_world = matrix(child.GetPrim()).copy()
            original_order = list(child.GetXformOpOrderAttr().Get())
            authored = stage.GetRootLayer().ExportToString()
            parking = DescendantParking(stage, camera)
            setters, _ = property_edits(
                stage,
                None,
                [{"id": "mark", "prim": str(child.GetPath()), "kind": "visibility"}],
                {"target": (target.GetPrim(),)},
                parking,
            )
            set_mark = setters["mark"]
            for _ in range(3):
                self.assertEqual(set_mark(False), "parked")
                self.assertEqual(child.GetVisibilityAttr().Get(), "inherited")
                parked = matrix(child.GetPrim())
                camera_world = UsdGeom.Xformable(camera.GetPrim()).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                )
                behind = camera_world.GetInverse().Transform(Gf.Vec3d(*parked[3, :3]))
                self.assertGreater(behind[2], 40)
                np.testing.assert_allclose(behind[:2], (0, 0), atol=1e-7)
                np.testing.assert_allclose(parked[:3, :3], original_world[:3, :3])
                self.assertEqual(set_mark(True), "inherited")
                np.testing.assert_allclose(matrix(child.GetPrim()), original_world)
                self.assertEqual(list(child.GetXformOpOrderAttr().Get()), original_order)
            parking.restore()
            np.testing.assert_allclose(matrix(child.GetPrim()), original_world)
            self.assertEqual(stage.GetRootLayer().ExportToString(), authored)

    def test_restore_before_bounds_and_variant_refresh(self):
        stage, target, child, camera = self.scene()
        child.GetVisibilityAttr().Set("inherited")
        parking = DescendantParking(stage, camera)
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        before = cache.ComputeWorldBound(target.GetPrim()).ComputeAlignedRange()
        parking.set(str(child.GetPath()), False)
        cache.Clear()
        parked = cache.ComputeWorldBound(target.GetPrim()).ComputeAlignedRange()
        self.assertGreater(parked.GetMax()[2], before.GetMax()[2] + 30)
        parking.restore()
        cache.Clear()
        restored = cache.ComputeWorldBound(target.GetPrim()).ComputeAlignedRange()
        np.testing.assert_allclose(restored.GetMin(), before.GetMin())
        np.testing.assert_allclose(restored.GetMax(), before.GetMax())

    def test_nontransformable_target_descendant_rejected(self):
        stage, target, _, camera = self.scene()
        scope = stage.DefinePrim("/World/Target/Scope", "Scope")
        with self.assertRaisesRegex(ValueError, "Xformable"):
            property_edits(
                stage,
                None,
                [{"id": "scope", "prim": str(scope.GetPath()), "kind": "visibility"}],
                {"target": (target.GetPrim(),)},
                DescendantParking(stage, camera),
            )

    def test_group_scope_reset_child_and_authored_op_collision(self):
        stage, target, _, camera = self.scene()
        scope = stage.DefinePrim("/World/Target/Group", "Scope")
        normal = UsdGeom.Cube.Define(stage, "/World/Target/Group/Normal")
        reset = UsdGeom.Cube.Define(stage, "/World/Target/Group/Reset")
        reset.SetResetXformStack(True)
        reset.AddTranslateOp().Set((0, 0, 2))
        authored_op = normal.AddTransformOp(opSuffix="harnessVisibilityPark")
        authored_op.Set(Gf.Matrix4d().SetTranslate((0.3, 0, 0)))
        originals = {str(p.GetPath()): matrix(p.GetPrim()).copy() for p in (normal, reset)}
        original_op = np.asarray(authored_op.Get()).copy()
        parking = DescendantParking(stage, camera)
        setters, _ = property_edits(
            stage,
            None,
            [{"id": "group", "prim": str(scope.GetPath()), "kind": "visibility"}],
            {"target": (target.GetPrim(),)},
            parking,
        )
        self.assertEqual(setters["group"](False), "parked")
        cw = UsdGeom.Xformable(camera.GetPrim()).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        for p in (normal, reset):
            behind = cw.GetInverse().Transform(Gf.Vec3d(*matrix(p.GetPrim())[3, :3]))
            self.assertGreater(behind[2], 40)
        parking.restore()
        for p in (normal, reset):
            np.testing.assert_allclose(matrix(p.GetPrim()), originals[str(p.GetPath())])
        np.testing.assert_allclose(authored_op.Get(), original_op)
        self.assertTrue(normal.GetPrim().GetAttribute("xformOp:transform:harnessVisibilityPark1"))

    def test_disarm_does_not_pin_later_geometry_order(self):
        stage, _, child, camera = self.scene()
        parking = DescendantParking(stage, camera)
        parking.set(str(child.GetPath()), False)
        parking.restore()
        stage.SetEditTarget(stage.GetRootLayer())
        child.AddScaleOp().Set((2, 2, 2))
        self.assertIn("xformOp:scale", [str(v) for v in child.GetXformOpOrderAttr().Get()])

    def test_disarm_preserves_prior_session_order(self):
        stage, _, child, camera = self.scene()
        prior = list(child.GetXformOpOrderAttr().Get())
        child.GetXformOpOrderAttr().Set(prior)
        parking = DescendantParking(stage, camera)
        parking.set(str(child.GetPath()), False)
        parking.restore()
        self.assertEqual(list(child.GetXformOpOrderAttr().Get()), prior)
        self.assertEqual(
            list(stage.GetSessionLayer().GetAttributeAtPath(
                child.GetPath().AppendProperty("xformOpOrder")
            ).default),
            prior,
        )

    def test_target_root_visibility_is_rejected(self):
        stage, target, _, camera = self.scene()
        with self.assertRaisesRegex(ValueError, "object pose"):
            property_edits(
                stage,
                None,
                [{"id": "target", "prim": str(target.GetPath()), "kind": "visibility"}],
                {"target": (target.GetPrim(),)},
                DescendantParking(stage, camera),
            )


if __name__ == "__main__":
    unittest.main()
