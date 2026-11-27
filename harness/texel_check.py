"""Check physical texel density and explicit tiling on selected USD meshes."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from PIL import Image
from pxr import Usd, UsdGeom, UsdShade

from harness.materials import source, value


class CheckError(ValueError):
    pass


def _texture(material):
    shader, _, _ = material.ComputeSurfaceSource()
    if not shader or shader.GetIdAttr().Get() != "UsdPreviewSurface":
        raise CheckError(f"{material.GetPath()}: expected UsdPreviewSurface")
    texture, channel = source(shader.GetInput("diffuseColor"))
    if not texture or texture.GetIdAttr().Get() != "UsdUVTexture" or channel != "rgb":
        raise CheckError(f"{material.GetPath()}: diffuseColor must use UsdUVTexture.outputs:rgb")
    if value(texture.GetInput("sourceColorSpace")) != "sRGB":
        raise CheckError(f"{texture.GetPath()}: expected explicit sRGB")
    reader, output = source(texture.GetInput("st"))
    if not reader or reader.GetIdAttr().Get() != "UsdPrimvarReader_float2" or output != "result":
        raise CheckError(f"{texture.GetPath()}: st must connect directly to a float2 primvar reader")
    name = value(reader.GetInput("varname"))
    if not isinstance(name, str) or not name:
        raise CheckError(f"{reader.GetPath()}: missing UV primvar name")
    asset = value(texture.GetInput("file"))
    if not asset or not asset.path:
        raise CheckError(f"{texture.GetPath()}: missing image")
    if "<UDIM>" in asset.path.upper() or not asset.resolvedPath:
        raise CheckError(f"{texture.GetPath()}: UDIM or unresolved image is unsupported")
    path = Path(asset.resolvedPath).resolve()
    try:
        with Image.open(path) as image:
            size = image.size
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise CheckError(f"{path}: {exc}") from exc
    return {"uv": name, "texture": str(texture.GetPath()), "image": str(path),
            "image_size": list(size), "sha256": digest,
            "wrapS": value(texture.GetInput("wrapS")), "wrapT": value(texture.GetInput("wrapT"))}


def _face_materials(prim, count):
    binding = UsdShade.MaterialBindingAPI(prim)
    base, _ = binding.ComputeBoundMaterial(UsdShade.Tokens.preview)
    result = [base] * count
    covered = set()
    for subset in binding.GetMaterialBindSubsets():
        if subset.GetElementTypeAttr().Get() != "face":
            raise CheckError(f"{subset.GetPath()}: only face material subsets are supported")
        faces = list(subset.GetIndicesAttr().Get() or [])
        if any(i < 0 or i >= count or i in covered for i in faces):
            raise CheckError(f"{subset.GetPath()}: invalid or overlapping faces")
        covered.update(faces)
        material, _ = UsdShade.MaterialBindingAPI(subset).ComputeBoundMaterial(UsdShade.Tokens.preview)
        if not material:
            raise CheckError(f"{subset.GetPath()}: missing bound material")
        for i in faces:
            result[i] = material
    if any(not material for material in result):
        raise CheckError(f"{prim.GetPath()}: faces without a bound material")
    return result


def _uvs(prim, name, point_count, corner_count, face_count):
    pv = UsdGeom.PrimvarsAPI(prim).FindPrimvarWithInheritance(name)
    if not pv:
        raise CheckError(f"{prim.GetPath()}: missing primvars:{name}")
    interpolation = pv.GetInterpolation()
    expected = {"vertex": point_count, "varying": point_count,
                "faceVarying": corner_count, "uniform": face_count, "constant": 1}.get(interpolation)
    values = pv.ComputeFlattened()
    if expected is None or values is None or len(values) != expected:
        raise CheckError(f"{pv.GetPath()}: unsupported or malformed UV interpolation {interpolation}")
    if not all(len(v) == 2 and all(math.isfinite(x) for x in v) for v in values):
        raise CheckError(f"{pv.GetPath()}: expected finite float2 UVs")
    return values, interpolation


def _density(positions, uvs, width, height):
    p0, p1, p2 = positions
    a = [p1[i] - p0[i] for i in range(3)]
    b = [p2[i] - p0[i] for i in range(3)]
    length = math.sqrt(sum(x*x for x in a))
    if length < 1e-12:
        raise CheckError("degenerate world-space triangle")
    parallel = sum(a[i]*b[i] for i in range(3)) / length
    perpendicular = math.sqrt(max(0.0, sum(x*x for x in b) - parallel*parallel))
    if perpendicular < 1e-12:
        raise CheckError("degenerate world-space triangle")
    d1 = [(uvs[1][i] - uvs[0][i]) * s / length for i, s in enumerate((width, height))]
    d2 = [((uvs[2][i] - uvs[0][i]) * s - parallel*d1[i]) / perpendicular
          for i, s in enumerate((width, height))]
    aa = sum(x*x for x in d1)
    bb = sum(x*x for x in d2)
    ab = sum(d1[i]*d2[i] for i in range(2))
    eigen_min = (aa + bb - math.sqrt(max(0.0, (aa-bb)**2 + 4*ab*ab))) / 2
    if eigen_min <= 1e-12:
        raise CheckError("collapsed UV triangle")
    return math.sqrt(eigen_min)


def inspect(stage, mesh_path, minimum=None):
    prim = stage.GetPrimAtPath(mesh_path)
    if not prim or not prim.IsA(UsdGeom.Mesh):
        raise CheckError(f"{mesh_path}: expected Mesh")
    mesh = UsdGeom.Mesh(prim)
    counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
    indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
    points = list(mesh.GetPointsAttr().Get() or [])
    valid, reason = mesh.ValidateTopology(indices, counts, len(points))
    if not valid or not counts or any(n < 3 for n in counts):
        raise CheckError(f"{mesh_path}: invalid topology: {reason}")
    materials = _face_materials(prim, len(counts))
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    matrix = cache.GetLocalToWorldTransform(prim)
    unit = UsdGeom.GetStageMetersPerUnit(stage)
    positions = [tuple(float(x)*unit for x in matrix.Transform(p)) for p in points]
    results = {}
    offset = 0
    for face, count in enumerate(counts):
        material = materials[face]
        key = str(material.GetPath())
        if key not in results:
            info = _texture(material)
            uv, interp = _uvs(prim, info["uv"], len(points), len(indices), len(counts))
            results[key] = {**info, "uv_values": uv, "interpolation": interp,
                            "densities": [], "s": [], "t": [], "faces": []}
        item = results[key]
        corners = indices[offset:offset+count]
        def uv_at(local):
            at = {"vertex": corners[local], "varying": corners[local],
                  "faceVarying": offset+local, "uniform": face, "constant": 0}[item["interpolation"]]
            return item["uv_values"][at]
        # Fan triangulation is exact for planar convex polygons; reject unsupported shapes.
        if count > 3:
            origin = positions[corners[0]]
            normal = None
            for j in range(1, count-1):
                a = [positions[corners[j]][k]-origin[k] for k in range(3)]
                b = [positions[corners[j+1]][k]-origin[k] for k in range(3)]
                cross = (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
                norm = math.sqrt(sum(x*x for x in cross))
                if norm < 1e-12:
                    raise CheckError(f"{mesh_path} face {face}: degenerate polygon")
                direction = tuple(x/norm for x in cross)
                if normal and sum(normal[k]*direction[k] for k in range(3)) < 1-1e-5:
                    raise CheckError(f"{mesh_path} face {face}: nonplanar or concave polygon unsupported")
                normal = direction
        for j in range(1, count-1):
            tri = (0, j, j+1)
            world = [positions[corners[k]] for k in tri]
            coords = [uv_at(k) for k in tri]
            try:
                item["densities"].append(_density(world, coords, *item["image_size"]))
            except CheckError as exc:
                raise CheckError(f"{mesh_path} face {face}: {exc}") from exc
        item["s"].extend(float(uv_at(k)[0]) for k in range(count))
        item["t"].extend(float(uv_at(k)[1]) for k in range(count))
        item["faces"].append(face)
        offset += count
    output = []
    errors = []
    for material, item in results.items():
        entry = {k: v for k, v in item.items() if k not in ("uv_values", "interpolation", "densities", "s", "t")}
        spans = [max(item[axis]) - min(item[axis]) for axis in ("s", "t")]
        entry["uv_range"] = [[min(item[axis]), max(item[axis])] for axis in ("s", "t")]
        entry["repeat_span"] = spans
        entry["min_texels_per_metre"] = min(item["densities"])
        entry["median_texels_per_metre"] = statistics.median(item["densities"])
        for axis, wrap in (("s", "wrapS"), ("t", "wrapT")):
            values = item[axis]
            if min(values) < -1e-8 or max(values) > 1+1e-8:
                if item[wrap] not in ("repeat", "mirror"):
                    errors.append(f"{material}: {wrap} must be explicitly repeat or mirror for out-of-range UVs (found {item[wrap]!r})")
        if minimum is not None and entry["min_texels_per_metre"] < minimum:
            errors.append(f"{material}: {entry['min_texels_per_metre']:.1f} texels/m below required {minimum:g}")
        output.append({"material": material, **entry})
    return {"mesh": mesh_path, "materials": output, "errors": errors, "ok": not errors}


def check(scene, meshes, minimum=None):
    stage = Usd.Stage.Open(str(scene))
    if not stage:
        raise CheckError(f"cannot open scene: {scene}")
    result = {"scene": str(Path(scene).resolve()), "meshes": [], "errors": []}
    for path in meshes:
        try:
            item = inspect(stage, path, minimum)
            result["meshes"].append(item)
            result["errors"].extend(item["errors"])
        except (CheckError, RuntimeError) as exc:
            result["meshes"].append({"mesh": path, "ok": False, "errors": [str(exc)]})
            result["errors"].append(str(exc))
    result["ok"] = not result["errors"]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--mesh", action="append", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-texels-per-metre", type=float)
    args = parser.parse_args(argv)
    if args.min_texels_per_metre is not None and args.min_texels_per_metre <= 0:
        parser.error("--min-texels-per-metre must be positive")
    try:
        report = check(args.scene, args.mesh, args.min_texels_per_metre)
    except CheckError as exc:
        report = {"scene": str(args.scene), "meshes": [], "errors": [str(exc)], "ok": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"texture scale: {'PASS' if report['ok'] else 'FAIL'} ({args.out})")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
