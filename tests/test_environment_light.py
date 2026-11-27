import hashlib
import json
from pathlib import Path

from pxr import Usd, UsdLux, UsdShade

from harness.campaign import load, prescribed, sample


ROOT = Path(__file__).resolve().parents[1] / "examples/campaign/warehouse-light"


def test_environment_light_composes_and_pairs():
    root, cfg = load(ROOT / "campaign.toml")
    stage = Usd.Stage.Open(str(root / cfg["stage"]))
    assert stage and str(stage.GetDefaultPrim().GetPath()) == "/World"
    assert stage.GetMetadata("metersPerUnit") == 1
    assert str(stage.GetMetadata("upAxis")) == "Z"
    for name in ("A", "B"):
        assert stage.GetPrimAtPath(f"/World/Packages/{name}")
    paper = UsdShade.Shader.Get(stage, "/World/PaperMaterial/Texture")
    assert [paper.GetInput(axis).Get() for axis in ("wrapS", "wrapT")] == ["repeat", "repeat"]
    dome = UsdLux.DomeLight.Get(stage, "/_HarnessWarehouseLight/Environment")
    assert dome
    assert dome.GetTextureFormatAttr().Get() == "latlong"
    assert dome.GetTextureFileAttr().Get().resolvedPath

    poses = prescribed(root, cfg)
    assert len(poses) == 4
    states = [sample(cfg, poses, 20261125, i) for i in range(4)]
    assert all(s["camera"] == states[0]["camera"] for s in states)
    assert all(s["objects"] == states[0]["objects"] for s in states)
    assert [(s["edits"]["environment"], s["edits"]["dome_gain"]) for s in states] == [
        ("small_workshop", 0.5),
        ("small_workshop", 1.0),
        ("university_workshop", 0.5),
        ("university_workshop", 1.0),
    ]
    assert all(s["edits"]["package_a_geometry"] == "cracker" for s in states)
    assert all(s["edits"]["package_b_geometry"] == "sugar" for s in states)

    manifest = json.loads((root / "assets/manifest.json").read_text())
    for item in manifest["files"]:
        path = root / "assets" / item["filename"]
        assert path.stat().st_size == item["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
