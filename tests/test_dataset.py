"""CPU checks of annotation alignment and deterministic pilot sampling."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from generate_dataset import resolve_roughness_shader, sample_frame, set_roughness, validate_frame
from capture_isaac import wait_for_annotators
from harness.dataset import _class_mask, _exclusive_box


def fixture_frame():
    rgb = np.zeros((12, 16, 4), dtype=np.uint8)
    rgb[:, 8:] = 50
    rgb[:, :, 3] = 255
    seg = np.zeros((12, 16), dtype=np.uint32)
    seg[2:6, 2:7] = 1
    seg[7:11, 10:15] = 2
    depth = np.ones((12, 16), dtype=np.float32)
    dtype = [("semanticId", "<u4"), ("x_min", "<i4"), ("y_min", "<i4"),
             ("x_max", "<i4"), ("y_max", "<i4"), ("occlusionRatio", "<f4")]
    boxes = np.array([(1, 2, 2, 6, 5, 0), (2, 10, 7, 14, 10, 0)], dtype=dtype)
    mapping = {"1": {"class": "carton"}, "2": {"class": "handheld_scanner"}}
    return rgb, seg, boxes, depth, mapping


def test_native_annotations_align():
    rgb, seg, native_boxes, depth, mapping = fixture_frame()
    bbox_mapping = {"11": {"class": "carton"}, "12": {"class": "handheld_scanner"}}
    native_boxes[0]["semanticId"] = 11
    native_boxes[1]["semanticId"] = 12
    boxes = validate_frame(rgb, seg, native_boxes, depth, mapping, bbox_mapping, 16, 12)
    assert [box["visible_pixels"] for box in boxes] == [20, 20]


def test_native_one_pixel_box_at_image_edge():
    rgb, seg, boxes, depth, mapping = fixture_frame()
    seg[7:11, 10:15] = 0
    seg[11, 15] = 2
    boxes[1]["x_min"] = boxes[1]["x_max"] = 15
    boxes[1]["y_min"] = boxes[1]["y_max"] = 11
    checked = validate_frame(rgb, seg, boxes, depth, mapping, mapping, 16, 12)
    assert checked[1]["xyxy_px"] == [15, 11, 15, 11]
    assert checked[1]["visible_pixels"] == 1
    assert _exclusive_box(checked[1]["xyxy_px"]) == [15, 11, 16, 12]


def test_training_class_mask_stable_when_replicator_ids_change():
    first = np.array([[0, 1, 2, 3, 4]], dtype=np.uint32)
    replay = np.array([[0, 1, 2, 4, 3]], dtype=np.uint32)
    first_labels = {"0": {"class": "BACKGROUND"}, "1": {"class": "UNLABELLED"},
                    "2": {"class": "carton"}, "3": {"class": "parts_tray"},
                    "4": {"class": "handheld_scanner"}}
    replay_labels = {**first_labels, "3": {"class": "handheld_scanner"},
                     "4": {"class": "parts_tray"}}
    expected = np.array([[0, 0, 1, 0, 2]], dtype=np.uint8)
    assert np.array_equal(_class_mask(first, first_labels), expected)
    assert np.array_equal(_class_mask(replay, replay_labels), expected)


def test_missing_or_shifted_label_fails():
    rgb, seg, boxes, depth, mapping = fixture_frame()
    seg[7:11, 10:15] = 0
    with pytest.raises(RuntimeError, match="overlap"):
        validate_frame(rgb, seg, boxes, depth, mapping, mapping, 16, 12)
    _, seg, boxes, _, _ = fixture_frame()
    boxes[0]["x_min"] = 8
    boxes[0]["x_max"] = 15
    with pytest.raises(RuntimeError, match="overlap"):
        validate_frame(rgb, seg, boxes, depth, mapping, mapping, 16, 12)


def test_seed_replays_exact_samples():
    first_rng = np.random.default_rng(17)
    second_rng = np.random.default_rng(17)
    first = [sample_frame(first_rng) for _ in range(2)]
    second = [sample_frame(second_rng) for _ in range(2)]
    assert first == second
    assert first[0] != sample_frame(np.random.default_rng(18))


def test_variant_roughness_uses_bound_material():
    from pxr import Usd

    stage = Usd.Stage.Open(str(Path(__file__).resolve().parents[1] / "examples/inspection/scene.usda"))
    variant = stage.GetPrimAtPath("/World/Props/Carton").GetVariantSet("surface_finish")
    variant.SetVariantSelection("worn")
    shader_path = resolve_roughness_shader(stage, "carton")
    assert shader_path == "/World/Props/Carton/Materials/Carton/Roughness"
    input_scale, readback = set_roughness(stage, shader_path, 0.9)
    assert readback["variant_base_scale"][0] == pytest.approx(0.85)
    assert readback["effective_scale"][0] == pytest.approx(0.765)
    assert readback["bias"][0] == pytest.approx(0.12)
    input_scale.GetAttr().Clear()


def test_waits_for_complete_paused_annotator_step():
    class App:
        def is_running(self):
            return True

        def update(self):
            pass

    class Orchestrator:
        count = 0

        def step(self, **kwargs):
            self.count += 1
            assert kwargs["delta_time"] == 0.0
            assert kwargs["pause_timeline"]
            assert kwargs["wait_for_render"]

    class Annotator:
        def __init__(self, orchestrator):
            self.orchestrator = orchestrator

        def get_data(self):
            return np.zeros((2, 3, 4), dtype=np.uint8) if self.orchestrator.count >= 3 else np.array([], dtype=np.uint8)

    orchestrator = Orchestrator()
    payloads, attempts = wait_for_annotators(
        App(), SimpleNamespace(orchestrator=orchestrator), {"rgb": Annotator(orchestrator)},
        SimpleNamespace(width=3, height=2, rt_subframes=8, renderer="PathTracing"),
    )
    assert attempts == 3
    assert payloads["rgb"].shape == (2, 3, 4)


def test_exported_camera_records_exposure_intrinsics_and_pose():
    from pxr import Usd, UsdGeom
    from capture_isaac import camera_model, configure_camera_exposure

    stage = Usd.Stage.Open(str(Path(__file__).resolve().parents[1] / 'examples/inspection/scene.usda'))
    camera = UsdGeom.Camera.Get(stage, '/World/Cameras/Overview')
    configure_camera_exposure(camera)
    model = camera_model(camera, 768, 512, UsdGeom.GetStageMetersPerUnit(stage))
    assert model['intrinsics_px']['fx'] == pytest.approx(896.0)
    assert model['authored_camera_exposure']['exposure:iso'] == 100
    assert len(model['camera_to_world_stage_units']) == 4
