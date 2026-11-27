"""Author a single metric, alpha textured mark on an explicit planar mesh quad."""

import math
import os
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt


def alpha_rectangles(image, cutoff=128):
    """Merge identical horizontal alpha spans across rows; return UV rectangles."""
    if type(cutoff) is not int or not 1 <= cutoff <= 255:
        raise ValueError("alpha cutoff must be an integer in [1,255]")
    alpha = np.asarray(image.convert("RGBA").getchannel("A")) >= cutoff
    height, width = alpha.shape
    active = {}
    rectangles = []
    for y in range(height + 1):
        row = alpha[y] if y < height else np.zeros(width, dtype=bool)
        changes = np.diff(np.r_[False, row, False].astype(np.int8))
        spans = set(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
        for span, start in list(active.items()):
            if span not in spans:
                rectangles.append((span[0] / width, 1 - y / height,
                                   span[1] / width, 1 - start / height))
                del active[span]
        for span in spans:
            active.setdefault(span, y)
    if not rectangles:
        raise ValueError("alpha image has no pixels above cutoff")
    return sorted(rectangles)


def add_mark(stage, mesh_path, face, owner_path, texture, width_m, height_m,
             center_uv=(0.5, 0.5), gap_m=0.0008, name="HandlingMark",
             keep_clear=(), alpha_cutoff=128):
    """Place a mark below its mesh, measuring dimensions and gap in world metres."""
    scalars = (width_m, height_m, gap_m, *center_uv)
    if (len(center_uv) != 2 or not all(math.isfinite(float(x)) for x in scalars)
            or not Sdf.Path.IsValidIdentifier(name) or not texture):
        raise ValueError("mark parameters must be finite and valid")
    if width_m <= 0 or height_m <= 0 or gap_m <= 0:
        raise ValueError("width, height and gap must be positive metres")
    texture_file = Path(texture)
    if not texture_file.is_absolute():
        layer_file = stage.GetRootLayer().realPath
        if not layer_file:
            raise ValueError("relative texture requires a file-backed stage")
        texture_file = Path(layer_file).parent / texture_file
    if not texture_file.is_file():
        raise ValueError(f"mark image missing: {texture_file}")
    with Image.open(texture_file) as image:
        rectangles = alpha_rectangles(image, alpha_cutoff)
    prim = stage.GetPrimAtPath(mesh_path)
    mesh = UsdGeom.Mesh(prim)
    if not mesh or not prim.IsValid():
        raise ValueError(f"mesh not found: {mesh_path}")
    owner = stage.GetPrimAtPath(owner_path)
    if not owner or not owner.IsValid() or not str(mesh_path).startswith(owner_path + "/"):
        raise ValueError("owner must be an ancestor of the target mesh")
    counts = mesh.GetFaceVertexCountsAttr().Get()
    if type(face) is not int or face < 0 or face >= len(counts) or counts[face] != 4:
        raise ValueError("select an existing quad face")
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    offset = sum(list(counts)[:face])
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    if not np.all(np.isfinite(points)):
        raise ValueError("mesh has non-finite points")
    units = UsdGeom.GetStageMetersPerUnit(stage)
    xform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    inverse = xform.GetInverse()
    if (not math.isfinite(units) or units <= 0 or
            not np.all(np.isfinite(np.asarray(xform))) or
            abs(xform.GetDeterminant()) < 1e-12):
        raise ValueError("invalid stage units or mesh transform")
    vertices = np.asarray([
        np.asarray(xform.Transform(Gf.Vec3d(*points[i]))) * units
        for i in indices[offset:offset + 4]
    ])
    origin = vertices[0]
    edge = vertices[1] - origin
    edge_len = np.linalg.norm(edge)
    if edge_len < 1e-8:
        raise ValueError("degenerate face")
    u = edge / edge_len
    normal = np.cross(vertices[1] - origin, vertices[2] - origin)
    normal_len = np.linalg.norm(normal)
    if normal_len < 1e-8:
        raise ValueError("degenerate face")
    normal /= normal_len
    if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
        normal = -normal
    if np.max(np.abs((vertices - origin) @ normal)) > 1e-5:
        raise ValueError("face is not planar")
    v = np.cross(normal, u)
    polygon = np.column_stack(((vertices - origin) @ u, (vertices - origin) @ v))
    def cross2(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    signed = [cross2(b - a, c - b) for a, b, c in
              zip(polygon, np.roll(polygon, -1, axis=0),
                  np.roll(polygon, -2, axis=0))]
    if (not np.all(np.isfinite(polygon)) or min(abs(s) for s in signed) < 1e-10
            or min(signed) * max(signed) <= 0):
        raise ValueError("face must be a finite convex quad")
    center = origin + center_uv[0] * (vertices[1] - origin) + center_uv[1] * (vertices[3] - origin)
    corners = [center + sx * width_m / 2 * u + sy * height_m / 2 * v
               for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    # Convex quad containment, including the requested edge clearance.
    projected = np.column_stack(((np.asarray(corners) - origin) @ u,
                                  (np.asarray(corners) - origin) @ v))
    orientation = np.sign(cross2(polygon[1] - polygon[0], polygon[2] - polygon[1]))
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        length = np.linalg.norm(b - a)
        if np.any(orientation * cross2(b - a, projected - a) / length < 1e-4):
            raise ValueError("mark crosses face boundary")
    if keep_clear:
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                  [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        mark_box = np.column_stack(((np.asarray(corners) - origin) @ u,
                                     (np.asarray(corners) - origin) @ v,
                                     np.full(4, gap_m)))
        mark_min = mark_box.min(axis=0)
        mark_max = mark_box.max(axis=0)
        mark_min[2] = -0.001
        mark_max[2] += 0.001
        for path in keep_clear:
            obstacle = stage.GetPrimAtPath(path)
            if not obstacle or not obstacle.IsValid():
                raise ValueError(f"keep-clear prim missing: {path}")
            bound = cache.ComputeWorldBound(obstacle).ComputeAlignedBox()
            if bound.IsEmpty():
                raise ValueError(f"keep-clear prim has no bounds: {path}")
            world = np.asarray([bound.GetCorner(i) for i in range(8)]) * units
            relative = world - origin
            obstacle_box = np.column_stack((relative @ u, relative @ v,
                                             relative @ normal))
            if np.all(mark_min <= obstacle_box.max(axis=0)) and np.all(
                    obstacle_box.min(axis=0) <= mark_max):
                raise ValueError(f"mark overlaps keep-clear prim: {path}")
    mark_path = mesh_path + "/" + name
    if stage.GetPrimAtPath(mark_path):
        raise ValueError(f"mark already exists: {mark_path}")
    mark = UsdGeom.Mesh.Define(stage, mark_path)
    vertices_out = []
    uv_out = []
    lower_left = corners[0] + gap_m * normal
    along = corners[1] - corners[0]
    upward = corners[3] - corners[0]
    for u0, v0, u1, v1 in rectangles:
        for su, sv in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
            world_point = lower_left + su * along + sv * upward
            local_point = inverse.Transform(Gf.Vec3d(*(world_point / units)))
            vertices_out.append(local_point)
            uv_out.append(Gf.Vec2f(su, sv))
    mark.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices_out, dtype=np.float32)))
    mark.CreateFaceVertexCountsAttr([4] * len(rectangles))
    mark.CreateFaceVertexIndicesAttr(list(range(4 * len(rectangles))))
    mark.CreateSubdivisionSchemeAttr("none")
    uv = UsdGeom.PrimvarsAPI(mark).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                UsdGeom.Tokens.faceVarying)
    uv.Set(uv_out)
    material = UsdShade.Material.Define(stage, owner_path + "/" + name + "Material")
    surface = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/PreviewSurface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    tex = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture)))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    st = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/ST")
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mark.GetPrim()).Bind(material)
    return mark


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="source USD stage")
    parser.add_argument("output", type=Path, help="new composing USD layer")
    parser.add_argument("--mesh", required=True, help="absolute mesh prim path")
    parser.add_argument("--face", required=True, type=int, help="quad face index")
    parser.add_argument("--owner", required=True, help="instance ancestor prim path")
    parser.add_argument("--texture", required=True, type=Path, help="RGBA image")
    parser.add_argument("--width-mm", required=True, type=float)
    parser.add_argument("--height-mm", required=True, type=float)
    parser.add_argument("--gap-mm", type=float, default=0.8)
    parser.add_argument("--center-u", type=float, default=0.5)
    parser.add_argument("--center-v", type=float, default=0.5)
    parser.add_argument("--keep-clear", action="append", default=[], metavar="PRIM",
                        help="reject overlap with this prim's bounds; repeat as needed")
    parser.add_argument("--alpha-cutoff", type=int, default=128,
                        help="hard cutout threshold in [1,255], default 128")
    args = parser.parse_args(argv)
    source, output, texture = (p.resolve(strict=True) for p in (args.source, args.output.parent, args.texture))
    output = output / args.output.name
    if output.exists() or output == source:
        parser.error("output must be a new USD file")
    source_stage = Usd.Stage.Open(str(source))
    if source_stage is None:
        parser.error("source is not a USD stage")
    layer = Sdf.Layer.CreateNew(str(output))
    layer.subLayerPaths = [os.path.relpath(source, output.parent)]
    stage = Usd.Stage.Open(layer)
    default = source_stage.GetDefaultPrim()
    if default:
        stage.SetDefaultPrim(stage.GetPrimAtPath(default.GetPath()))
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source_stage))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source_stage))
    add_mark(stage, args.mesh, args.face, args.owner,
             os.path.relpath(texture, output.parent), args.width_mm / 1000,
             args.height_mm / 1000, (args.center_u, args.center_v),
             args.gap_mm / 1000, keep_clear=args.keep_clear,
             alpha_cutoff=args.alpha_cutoff)
    layer.Save()
    print(output)


if __name__ == "__main__":
    main()
