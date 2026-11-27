# /// script
# requires-python = "==3.11.*"
# dependencies = ["bpy==4.5.17"]
# ///
"""CPU path tracing via Blender's USD importer. Invoked in an isolated process."""

import json
import re
import sys
from pathlib import Path

import bpy


def usd_path(obj):
    # Blender disambiguates names across the entire file; USD names are sibling-local.
    names = []
    while obj:
        names.append(re.sub(r"\.\d{3,}$", "", obj.name))
        obj = obj.parent
    return "/" + "/".join(reversed(names))


def main(job_path):
    job = json.loads(job_path.read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    status = bpy.ops.wm.usd_import(
        filepath=job["stage"],
        import_usd_preview=True,
        mtl_purpose="MTL_PREVIEW",
        import_guide=False,
        import_proxy=False,
        import_render=True,
        import_visible_only=True,
        merge_parent_xform=False,
        light_intensity_scale=1.0,
        apply_unit_conversion_scale=True,
    )
    if status != {"FINISHED"}:
        raise RuntimeError(f"Blender USD import failed: {status}")
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = job["samples"]
    scene.cycles.seed = 19
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    for name, view in job["views"].items():
        cameras = [obj for obj in scene.objects if obj.type == "CAMERA" and usd_path(obj) == view["camera"]]
        if len(cameras) != 1:
            raise ValueError(f"Could not uniquely resolve imported camera {view['camera']}")
        scene.camera = cameras[0]
        scene.render.resolution_x, scene.render.resolution_y = view["size"]
        scene.render.filepath = str(job_path.parent / (name + ".png"))
        if bpy.ops.render.render(write_still=True) != {"FINISHED"}:
            raise RuntimeError(f"Cycles failed to render {name}")
    result = {
        "renderer": "Blender Cycles CPU",
        "version": bpy.app.version_string,
        "samples": scene.cycles.samples,
        "seed": scene.cycles.seed,
        "denoiser": scene.cycles.denoiser,
        "max_bounces": scene.cycles.max_bounces,
        "view_transform": scene.view_settings.view_transform,
        "look": scene.view_settings.look,
        "display_device": scene.display_settings.display_device,
        "display_exposure": scene.view_settings.exposure,
        "light_import_scale": 1,
    }
    (job_path.parent / "renderer.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
