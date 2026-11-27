import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Usd, UsdGeom

from harness.author import initialize
from harness.campaign import digest, inspect_frame, load, read, sample, verify, write
from harness.qualification import release, replay


class NewScene(unittest.TestCase):
    def test_onboard_new_components_and_independent_shards(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            source = d / "cell.usda"
            stage = Usd.Stage.CreateNew(str(source))
            UsdGeom.SetStageMetersPerUnit(stage, 1)
            UsdGeom.SetStageUpAxis(stage, "Z")
            world = UsdGeom.Xform.Define(stage, "/Optics")
            stage.SetDefaultPrim(world.GetPrim())
            for name, x in [("Lens1", 0), ("Lens2", 3)]:
                prim = UsdGeom.Sphere.Define(stage, "/Optics/" + name)
                prim.AddTranslateOp().Set((x, 0, 1))
            stage.GetRootLayer().Save()
            before = digest(source)
            result = initialize(
                source,
                d / "optic-project",
                [("lens", "/Optics/Lens1"), ("lens", "/Optics/Lens2")],
                640,
                480,
            )
            root, cfg = load(result)
            self.assertEqual(cfg["classes"], {"lens": 1})
            wrapped = Usd.Stage.Open(str(root / cfg["stage"]))
            self.assertTrue(wrapped.GetPrimAtPath("/Optics/Lens1"))
            self.assertTrue(wrapped.GetPrimAtPath("/_HarnessStudio/Key"))
            self.assertEqual(cfg["layouts"][0]["anchors"]["object2"], [3, 0, 0])
            self.assertEqual(digest(source), before)
            all_frames = [sample(cfg, None, 123, i) for i in range(9)]
            self.assertEqual(
                all_frames[6:], [sample(cfg, None, 123, i) for i in range(6, 9)]
            )
            self.assertNotEqual(all_frames[0], sample(cfg, None, 124, 0))
            with self.assertRaises(FileExistsError):
                initialize(source, root, [("lens", "/Optics/Lens1")])
            with self.assertRaises(ValueError):
                initialize(source, d / "absent", [("lens", "/World/Bad")])
            self.assertFalse((d / "absent").exists())


def fixture(root):
    root.mkdir()
    head = {
        "campaign_key": "known",
        "resolution": [5, 4],
        "instances": [
            {"class": "component", "id": "first"},
            {"class": "component", "id": "second"},
        ],
        "classes": {"component": 7},
    }
    write(root / "campaign.json", head)
    folder = root / "frames" / "00000000"
    folder.mkdir(parents=True)
    inst = np.zeros((4, 5), np.uint16)
    inst[1:3, 1:3] = 1
    inst[2:, 4] = 2
    Image.fromarray(inst).save(folder / "instances.png")
    Image.fromarray(np.uint8(inst > 0) * 7).save(folder / "classes.png")
    rgb = np.repeat(np.arange(20, dtype=np.uint8).reshape(4, 5, 1) * 9, 3, axis=2)
    Image.fromarray(rgb).save(folder / "rgb.png")
    np.savez_compressed(folder / "range_m.npz", range_m=np.ones((4, 5), np.float32))
    summary = {
        "campaign_key": "known",
        "index": 0,
        "sample": {
            "camera": {"id": "fixture"},
            "layout": "a",
            "edits": {"wall": "metal"},
        },
        "readback": {},
        "sha256": {p.name: digest(p) for p in folder.iterdir()},
        "objects": [
            {
                "id": "first",
                "category_id": 7,
                "instance_id": 1,
                "class": "component",
                "bbox_xywh": [1, 1, 2, 2],
                "area": 4,
            },
            {
                "id": "second",
                "category_id": 7,
                "instance_id": 2,
                "class": "component",
                "bbox_xywh": [4, 2, 1, 2],
                "area": 2,
            },
        ],
    }
    write(folder / "annotation.json", summary)
    return head, folder


class CampaignFiles(unittest.TestCase):
    def test_masks_ids_export_and_missing_frame(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t)
            root = path / "native"
            head, frame = fixture(root)
            out = path / "export"
            report = verify(root, out, 0, 1, coco=True)
            self.assertEqual(report["boxes"], 2)
            coco = read(out / "coco.json")
            for obj in coco["annotations"]:
                encoded = obj["segmentation"]
                decoded = np.repeat(
                    np.arange(len(encoded["counts"])) % 2, encoded["counts"]
                ).reshape(encoded["size"], order="F")
                self.assertEqual(int(decoded.sum()), obj["area"])
                self.assertEqual(
                    np.array_equal(
                        decoded > 0,
                        np.array(Image.open(frame / "instances.png")) == obj["id"],
                    ),
                    True,
                )
            idx = json.loads((out / "training.jsonl").read_text())
            self.assertTrue((out / idx["image"]).is_file())
            with self.assertRaises(OSError):
                verify(root, out, 0, 2)
            row = read(frame / "annotation.json")
            row["objects"][1]["bbox_xywh"] = [0, 0, 4, 2]
            write(frame / "annotation.json", row)
            with self.assertRaises(ValueError):
                inspect_frame(frame, head)

    def test_release_fails_closed_and_replay_verifies_content(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t)
            root = path / "native"
            fixture(root)
            second = path / "second"
            fixture(second)
            self.assertEqual(
                replay(root, second, 0, 1, 0.01, 0.99, 0.002)["status"], "pass"
            )
            policy = path / "policy.toml"
            policy.write_text(
                "minimum_frames = 1\nrequired_visual_frames = [0]\nrequire_replay = true\n"
            )
            self.assertEqual(release(root, policy, path / "release")["status"], "fail")
            (second / "frames/00000000/rgb.png").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                replay(root, second, 0, 1, 0.01, 0.99, 0.002)


class DatasetControls(unittest.TestCase):
    def test_registered_gate_requires_every_promised_photo_and_threshold_file(self):
        from harness.qualification import registered_inputs

        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            spec = {"poses": "poses.json", "quality_policy": "limits.toml"}
            write(root / "poses.json", [{"layout": "room", "reference": "shot", "rgb_sha256": "abc"}])
            limits = root / "limits.toml"
            limits.write_text("limit = 0.1\n")
            row = {"scene": "room", "reference": "shot", "real_sha256": {"rgb.png": "abc"}}
            report = {"frames": [row], "config_sha256": digest(limits)}
            self.assertTrue(registered_inputs(root, spec, report)[1])
            for frames in ([], [row, row], [dict(row, reference="different")],
                           [dict(row, real_sha256={"rgb.png": "unrelated"})]):
                self.assertFalse(registered_inputs(root, spec, dict(report, frames=frames))[1])
            limits.write_text("limit = 0.9\n")
            self.assertFalse(registered_inputs(root, spec, report)[1])

    def test_sampling_coverage_is_planned_not_rendered(self):
        from harness.survey import survey

        report = survey(
            Path(__file__).resolve().parents[1] / "examples/campaign/production.toml",
            16,
            20261125,
        )
        self.assertEqual(report["rendered_frames"], 0)
        self.assertEqual(len(report["camera_by_layout"]), 4)
        self.assertTrue(
            all(
                sorted(a.values()) == [1, 1, 1, 1]
                for a in report["camera_by_layout"].values()
            )
        )

    def test_execute_retries_then_verifies_saved_shard_and_refuses_overlap(self):
        from harness.shards import execute, validate_plan

        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            cfg = d / "recipe.toml"
            cfg.write_text("# fixture\n")
            plan = {
                "status": "planned",
                "config_sha256": digest(cfg),
                "seed": 42,
                "shards": [{"start": 0, "count": 1}],
                "frames": 1,
            }
            p = d / "plan.json"
            write(p, plan)
            out = d / "native"
            head, folder = fixture(out)
            write(out / "campaign.json", {**head, "seed": 42})
            calls = []

            def runner(cmd, **kwargs):
                calls.append(cmd)
                return 1 if len(calls) == 1 else 0

            result = read(execute(cfg, p, out, runner=runner))
            self.assertEqual(result["status"], "worker_complete")
            self.assertEqual(result["shards"][0]["verified_frames"], 1)
            self.assertEqual(len(calls), 2)
            with self.assertRaises(ValueError):
                validate_plan({**plan, "shards": [{"start": 1, "count": 1}]}, cfg)
            with self.assertRaises(ValueError):
                execute(cfg, p, out, render_args=["--seed", "9"], runner=runner)

    def test_unlabelled_visible_surface_is_detected_in_range(self):
        from harness.visibility import depth_witness

        fx = 80
        yy, xx = np.indices((24, 24))
        depths = np.sqrt(1 + ((xx - 12) / fx) ** 2 + ((yy - 12) / fx) ** 2).astype(
            np.float32
        )
        mesh = np.c_[
            ((xx[6:18, 6:18] - 12) / fx).flatten(),
            (-(yy[6:18, 6:18] - 12) / fx).flatten(),
            -np.ones(144),
        ]
        camera = {"K": [fx, 0, 12, 0, fx, 12, 0, 0, 1], "matrix_row": np.eye(4)}
        self.assertEqual(depth_witness(mesh, np.eye(4), camera, depths), 144)
        self.assertEqual(depth_witness(mesh, np.eye(4), camera, depths * 0.5), 0)

    def test_gain_cannot_be_evaluated_on_calibration_view(self):
        from harness.holdout import apply

        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            poses = root / "poses.json"
            fit = root / "freeze.json"
            out = root / "result.json"
            write(fit, {"chosen_gain": 0.125, "calibration_references": ["a"]})
            write(poses, [{"reference": "a"}])
            with self.assertRaises(ValueError):
                apply(poses, fit, ["key"], out)
            self.assertFalse(out.exists())
            write(poses, [{"reference": "b"}])
            apply(poses, fit, ["key", "fill"], out)
            self.assertEqual(read(out)[0]["edits"], {"key": 0.125, "fill": 0.125})


def test_existing_usd_semantics_can_initialize_without_custom_names():
    from harness.author import labelled_targets, target_argument
    from pxr import Sdf

    assert target_argument("光学組立品=/Factory/Optics") == (
        "光学組立品",
        "/Factory/Optics",
    )
    with tempfile.TemporaryDirectory() as t:
        src = Path(t) / "labels.usda"
        stage = Usd.Stage.CreateNew(str(src))
        for path in ("/Cell/One", "/Cell/One/Geometry", "/Cell/Two"):
            prim = UsdGeom.Xform.Define(stage, path).GetPrim()
            prim.AddAppliedSchema("SemanticsLabelsAPI:class")
            prim.CreateAttribute(
                "semantics:labels:class", Sdf.ValueTypeNames.TokenArray
            ).Set(["housing"])
        stage.GetRootLayer().Save()
        assert labelled_targets(src) == [
            ("housing", "/Cell/One"),
            ("housing", "/Cell/Two"),
        ]
