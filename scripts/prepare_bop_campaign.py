"""Convert BOP textured meshes and registered real RGB-D frames into portable campaign inputs.

Use with the BOP YCB-Video models, base and test_bop19 archives. Mesh positions in
BOP millimetres are converted once. RGB test pixels are used only for paired evaluation.
"""

import argparse
import json
from hashlib import file_digest, sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

NAMES = {
    1: "master_chef_can",
    2: "cracker_box",
    3: "sugar_box",
    4: "soup_can",
    5: "mustard_bottle",
    6: "tuna_can",
    10: "banana",
    12: "cleanser",
    15: "power_drill",
    17: "scissors",
}


def material(stage, path, image, roughness):
    mat = UsdShade.Material.Define(stage, path)
    surf = UsdShade.Shader.Define(stage, path + "/Surface")
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    uv = UsdShade.Shader.Define(stage, path + "/UV")
    uv.CreateIdAttr("UsdPrimvarReader_float2")
    uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture = UsdShade.Shader.Define(stage, path + "/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(image))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        uv.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    )
    surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )
    mat.CreateSurfaceOutput().ConnectToSource(
        surf.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    return mat


def convert(z, n, out):
    stem = f"obj_{n:06d}"
    ply = z.read(f"models/{stem}.ply")
    atlas = z.read(f"models/{stem}.png")
    header, data = ply.split(b"end_header", 1)
    lines = header.decode().splitlines()
    if "format ascii 1.0" not in lines:
        raise ValueError("expected the BOP ASCII textured PLY")
    nv = int(next(l.split()[-1] for l in lines if l.startswith("element vertex ")))
    nf = int(next(l.split()[-1] for l in lines if l.startswith("element face ")))
    stream = BytesIO(data.lstrip())
    v = np.loadtxt(stream, max_rows=nv, dtype=np.float32)
    faces = np.loadtxt(stream, dtype=np.int32)
    if v.shape != (nv, 8) or faces.shape != (nf, 4) or not np.all(faces[:, 0] == 3):
        raise ValueError(f"unsupported BOP mesh schema: {stem}")
    target = out / "models"
    target.mkdir(exist_ok=True)
    im = Image.open(BytesIO(atlas)).convert("RGB")
    im.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    im.save(target / (stem + ".jpg"), quality=94, subsampling=0)
    s = Usd.Stage.CreateNew(str(target / (stem + ".usdc")))
    UsdGeom.SetStageMetersPerUnit(s, 1.0)
    UsdGeom.SetStageUpAxis(s, "Z")
    root = UsdGeom.Xform.Define(s, "/Object")
    s.SetDefaultPrim(root.GetPrim())
    m = UsdGeom.Mesh.Define(s, "/Object/Mesh")
    xyz = np.ascontiguousarray(v[:, :3] * 0.001)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(xyz))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(nf, 3, dtype=np.int32)))
    m.CreateFaceVertexIndicesAttr(
        Vt.IntArray.FromNumpy(np.ascontiguousarray(faces[:, 1:].ravel()))
    )
    m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(v[:, 3:6])))
    m.SetNormalsInterpolation("vertex")
    st = UsdGeom.PrimvarsAPI(m).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(v[:, 6:8])))
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateExtentAttr(m.ComputeExtent(m.GetPointsAttr().Get()))
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(
        material(s, "/Object/Material", stem + ".jpg", 0.53)
    )
    s.GetRootLayer().Save()
    return xyz, {
        "id": n,
        "ply_sha256": sha256(ply).hexdigest(),
        "source_atlas_sha256": sha256(atlas).hexdigest(),
        "vertices": nv,
        "faces": nf,
        "output_atlas_size": list(im.size),
    }


def pose(R, t):
    m = np.eye(4)
    m[:3, :3] = np.array(R).reshape(3, 3)
    m[:3, 3] = np.array(t) * 0.001
    return m


def table_fit(depth, mask, cam, world_from_cv, guess):
    """Fit table z=ax+by+c in measured points near the annotated objects' supports."""
    ys, xs = np.where(~mask & (depth > 0))
    ys, xs = ys[::11], xs[::11]
    K = np.array(cam["cam_K"]).reshape(3, 3)
    rays = np.linalg.solve(K, np.vstack((xs, ys, np.ones_like(xs))))
    z = depth[ys, xs] * cam["depth_scale"] * 0.001
    pw = (world_from_cv[:3, :3] @ (rays * z) + world_from_cv[:3, 3:4]).T
    candidates = pw[
        (np.abs(pw[:, 2] - guess) < 0.085)
        & (np.abs(pw[:, 0]) < 1.0)
        & (np.abs(pw[:, 1]) < 1.0)
    ]
    if len(candidates) < 80:
        raise ValueError("insufficient table depth near the objects")
    edges = np.arange(guess - 0.09, guess + 0.095, 0.003)
    hist, _ = np.histogram(candidates[:, 2], edges)
    n = int(np.argmax(hist))
    mode = (edges[n] + edges[n + 1]) / 2
    near = candidates[np.abs(candidates[:, 2] - mode) < 0.009]
    design = np.column_stack((near[:, :2], np.ones(len(near))))
    coeff = np.linalg.lstsq(design, near[:, 2], rcond=None)[0]
    residual = design @ coeff - near[:, 2]
    return {
        "plane_z_equals_ax_by_c": coeff.tolist(),
        "inlier_points": len(near),
        "rms_m": float(np.sqrt(np.mean(residual**2))),
    }


def make_stage(out, reference_scene):
    s = Usd.Stage.CreateNew(str(out / "paired.usda"))
    UsdGeom.SetStageMetersPerUnit(s, 1.0)
    UsdGeom.SetStageUpAxis(s, "Z")
    world = UsdGeom.Xform.Define(s, "/World")
    s.SetDefaultPrim(world.GetPrim())
    UsdGeom.Camera.Define(s, "/World/Camera")
    l = UsdLux.RectLight.Define(s, "/World/Key")
    l.CreateIntensityAttr(2700)
    l.CreateWidthAttr(0.8)
    l.CreateHeightAttr(0.7)
    l.AddTranslateOp().Set((-0.3, -0.4, 1.2))
    d = UsdLux.DomeLight.Define(s, "/World/Fill")
    d.CreateIntensityAttr(300.0)
    import os

    imgs = list((reference_scene / "lighting").glob("*.hdr")) + list(
        (reference_scene / "lighting").glob("*.exr")
    )
    if imgs:
        d.CreateTextureFileAttr(Sdf.AssetPath(os.path.relpath(imgs[0], out)))
    m = UsdGeom.Mesh.Define(s, "/World/Table")
    m.CreateFaceVertexCountsAttr([4])
    m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    m.CreatePointsAttr([(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0)])
    m.CreateSubdivisionSchemeAttr("none")
    uv = UsdGeom.PrimvarsAPI(m).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    uv.Set([(0, 0), (4, 0), (4, 4), (0, 4)])
    texture = next((reference_scene / "textures").glob("*wood*base.jpg"), None)
    if texture:
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(
            material(s, "/World/TableMaterial", os.path.relpath(texture, out), 0.66)
        )
    s.GetRootLayer().Save()


def file_sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("models", "test", "base", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--reference-scene", type=Path, default=Path("examples/inspection"))
    p.add_argument("--scenes", nargs="+", type=int, default=[50, 51])
    p.add_argument("--views", type=int, default=3)
    p.add_argument(
        "--ranks",
        nargs="+",
        type=float,
        help="Explicit normalized positions in each sorted set of RGB keyframes",
    )
    a = p.parse_args()
    ranks = a.ranks if a.ranks is not None else np.linspace(0, 1, a.views).tolist()
    if not ranks or any(not 0 <= v <= 1 for v in ranks):
        p.error("ranks must lie in [0,1]")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    test = ZipFile(a.test)
    models = ZipFile(a.models)
    base = ZipFile(a.base)
    (out / "YCB-LICENSE.txt").write_bytes(
        base.read("ycbv/dataset_info.md").split(b"## License")[-1].strip() + b"\n"
    )
    allids = {2, 3, 5}
    jobs = []
    for sc in a.scenes:
        prefix = f"test/{sc:06d}/"
        ims = sorted(
            int(Path(n).stem)
            for n in test.namelist()
            if n.startswith(prefix + "rgb/") and n.endswith(".png")
        )
        cams = json.loads(test.read(prefix + "scene_camera.json"))
        gts = json.loads(test.read(prefix + "scene_gt.json"))
        indices = [int(v * (len(ims) - 1)) for v in ranks]
        if len(indices) != len(set(indices)):
            raise ValueError("duplicate keyframe ranks")
        for i in indices:
            fr = ims[i]
            gt = gts[str(fr)]
            allids.update(o["obj_id"] for o in gt)
            jobs.append((sc, fr, cams[str(fr)], gt))
    assets = {}
    srcs = []
    for n in sorted(allids):
        pts, rec = convert(models, n, out)
        assets[n] = pts
        srcs.append(rec)
    make_stage(out, a.reference_scene.resolve())
    rows = []
    ref = out / "real"
    ref.mkdir()
    for sc, fr, cam, objs in jobs:
        sid = f"{sc:06d}_{fr:06d}"
        prefix = f"test/{sc:06d}/"
        root = ref / sid
        root.mkdir()
        rgb = test.read(prefix + f"rgb/{fr:06d}.png")
        (root / "rgb.png").write_bytes(rgb)
        raw = test.read(prefix + f"depth/{fr:06d}.png")
        (root / "z-depth.png").write_bytes(raw)
        measured = np.asarray(Image.open(BytesIO(raw)))
        h, w = measured.shape
        label = np.zeros((h, w), np.uint8)
        T = pose(cam["cam_R_w2c"], cam["cam_t_w2c"])
        W = np.linalg.inv(T)
        state = {f"obj{n}": {"visible": False} for n in sorted(allids)}
        bottom = []
        for j, o in enumerate(objs):
            n = o["obj_id"]
            m = W @ pose(o["cam_R_m2c"], o["cam_t_m2c"])
            state[f"obj{n}"] = {"visible": True, "matrix_row": m.T.tolist()}
            vis = (
                np.asarray(
                    Image.open(
                        BytesIO(test.read(prefix + f"mask_visib/{fr:06d}_{j:06d}.png"))
                    )
                )
                != 0
            )
            label[vis] = n
            bottom.append(float((assets[n] @ m[:3, :3].T + m[:3, 3])[:, 2].min()))
        Image.fromarray(label).save(root / "classes.png")
        fit = table_fit(measured, label > 0, cam, W, np.quantile(bottom, 0.20))
        rows.append(
            {
                "layout": f"real_scene_{sc}",
                "reference": sid,
                "table_fit": fit,
                "camera": {
                    "id": sid,
                    "K": cam["cam_K"],
                    "matrix_row": (W @ np.diag([1, -1, -1, 1])).T.tolist(),
                },
                "objects": state,
                "depth_scale_mm": cam["depth_scale"],
                "rgb_sha256": sha256(rgb).hexdigest(),
            }
        )
    (out / "poses.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        'stage = "paired.usda"',
        'poses = "poses.json"',
        'camera_prim = "/World/Camera"',
        'table_mesh = "/World/Table"',
        f"width = {w}",
        f"height = {h}",
        "[classes]",
    ]
    for n in sorted(allids):
        lines.append(f"{NAMES.get(n, 'object_' + str(n))} = {n}")
    for n in sorted(allids):
        lines += [
            "",
            "[[instances]]",
            f'id = "obj{n}"',
            f'class = "{NAMES.get(n, "object_" + str(n))}"',
            f'prim = "/World/Props/Obj{n}"',
            f'asset = "models/obj_{n:06d}.usdc"',
        ]
    (out / "paired.toml").write_text("\n".join(lines) + "\n")
    (out / "sources.json").write_text(
        json.dumps(
            {
                "dataset": "BOP YCB-Video",
                "snapshot": "5c2c4aa229800355648cd268040aa814f8dc94f0",
                "archive_base_url": "https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/5c2c4aa229800355648cd268040aa814f8dc94f0/",
                "archives": {p.name: file_sha(p) for p in (a.models, a.test, a.base)},
                "converted_models": srcs,
                "real_scenes": a.scenes,
                "keyframe_ranks": ranks,
                "units": "BOP PLY/pose translations mm; authored USD metres",
                "appearance": "photographic scanned albedo; constant roughness is an estimate",
                "license": "YCB-LICENSE.txt",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{len(rows)} paired physical views, {len(assets)} textured objects: {out}")


if __name__ == "__main__":
    main()
