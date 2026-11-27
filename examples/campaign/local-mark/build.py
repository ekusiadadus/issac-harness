"""Build a portable production campaign with one localized carton mark."""

from pathlib import Path
import json
import sys

from pxr import Sdf, Usd, UsdGeom

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from harness.surface_mark import add_mark
from harness.finishes import bake


CAMPAIGN = HERE.parent
STAGE = HERE / "scene.usda"
CONFIG = HERE / "production.toml"
COMPARE = HERE / "compare.toml"
COMPARE_POSES = HERE / "compare-poses.json"


def build():
    layer = Sdf.Layer.CreateNew(str(STAGE)) if not STAGE.exists() else Sdf.Layer.FindOrOpen(str(STAGE))
    layer.Clear()
    layer.subLayerPaths = [str(Path("../../inspection/scene.usda"))]
    stage = Usd.Stage.Open(layer)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    add_mark(stage, "/World/Props/Carton/WallFront", 31, "/World/Props/Carton",
             "../handling-mark.png", 0.060, 0.037, (0.17, 0.40), 0.0008,
             keep_clear=("/World/Props/Carton/LabelPrint",))
    stage.GetRootLayer().Save()

    config = (CAMPAIGN / "production.toml").read_text()
    config = config.replace('stage = "../inspection/scene.usda"', 'stage = "scene.usda"')
    config = config.replace('asset = "ycb/', 'asset = "../ycb/')
    bake(HERE / "paper.toml", HERE / "finishes")
    config += '''\n[[edits]]
id = "mark_visible"
prim = "/World/Props/Carton/WallFront/HandlingMark"
kind = "visibility"
choices = [false, true]
'''
    CONFIG.write_text(config)
    compare = config.replace('height = 720\n',
                             'height = 720\nposes = "compare-poses.json"\n', 1)
    COMPARE.write_text(compare)
    objects = {
        "crate": {"bottom_xyz_m": [0.0, 0.0, 0.8925], "yaw_deg": 0.0,
                  "visible": True},
    }
    for ident in ("bin", "device", "crackers", "sugar", "mustard"):
        objects[ident] = {"bottom_xyz_m": [0.0, 0.0, 0.8925],
                          "yaw_deg": 0.0, "visible": False}
    edits = {"key": 1.0, "side": 1.0, "fill": 1.0, "dome": 1.0,
             "bench": "/World/Materials/Wood",
             "wall": "/World/Materials/Wall_Paint",
             "polymer": "standard", "carton_tape": True,
             "foam_pad": False, "paper": "finish07"}
    camera = {"id": "fixed_front", "eye_m": [0.0, -0.6, 1.05],
              "target_m": [0.0, -0.13, 0.96],
              "K": [1200.0, 0.0, 480.0, 0.0, 1200.0, 360.0,
                    0.0, 0.0, 1.0]}
    poses = [{"layout": "fixed_carton_front", "camera": camera,
              "objects": objects, "edits": {**edits, "mark_visible": visible}}
             for visible in (False, True, False)]
    COMPARE_POSES.write_text(json.dumps(poses, indent=2) + "\n")
    return CONFIG


if __name__ == "__main__":
    print(build())
