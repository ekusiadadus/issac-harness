"""Inspect the portable surface terminal actually bound to rendered geometry."""

import math
from pathlib import Path, PureWindowsPath

from PIL import Image
from pxr import Gf, UsdGeom, UsdShade


def portable_path(value):
    return bool(value) and not (
        "://" in value or Path(value).is_absolute() or PureWindowsPath(value).drive or "\\" in value
    )


def source(port):
    if port:
        attrs = port.GetValueProducingAttributes(shaderOutputsOnly=True)
        if len(attrs) == 1:
            attr = attrs[0]
            return UsdShade.Shader(attr.GetPrim()), attr.GetBaseName()
    return None, None


def value(port):
    if port:
        attrs = port.GetValueProducingAttributes()
        if len(attrs) == 1:
            return attrs[0].Get()
    return None


def uv_name(texture):
    node, output = source(texture.GetInput("st"))
    visited = set()
    while node and output == "result" and node.GetPath() not in visited:
        visited.add(node.GetPath())
        shader_id = node.GetIdAttr().Get()
        if shader_id == "UsdPrimvarReader_float2":
            return value(node.GetInput("varname"))
        if shader_id != "UsdTransform2d":
            break
        node, output = source(node.GetInput("in"))
    return None


def image_info(path, channel, root, cache, significant_channels=None):
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"texture outside the scene package: {path}")
    if path not in cache:
        with Image.open(path) as image:
            image.load()
            ranges = image.convert("RGBA").getextrema()
            varying_channels = {c for c, (lo, hi) in zip("rgba", ranges) if lo != hi}
            channel_ranges = dict(zip("rgba", ranges))
            rgba = image.convert("RGBA")
            histograms = {c: rgba.getchannel(c.upper()).histogram() for c in "rgba"}
            def percentile(histogram, fraction):
                target = sum(histogram) * fraction
                cumulative = 0
                for position, count in enumerate(histogram):
                    cumulative += count
                    if cumulative >= target:
                        return position
                return 255
            robust_ranges = {
                c: percentile(hist, 0.95) - percentile(hist, 0.05)
                for c, hist in histograms.items()
            }
            cache[path] = (
                {
                    "file": path.relative_to(root).as_posix(),
                    "size": list(image.size),
                    "mode": image.mode,
                },
                varying_channels,
                channel_ranges,
                robust_ranges,
            )
    info, varying_channels, channel_ranges, robust_ranges = cache[path]
    variation = max((channel_ranges[c][1] - channel_ranges[c][0] for c in (channel or "") if c in channel_ranges), default=0)
    significant = significant_channels or channel or ""
    robust = max((robust_ranges[c] for c in significant if c in robust_ranges), default=0)
    return {**info, "varies": bool(varying_channels.intersection(channel or "")), "range": variation, "p95_p05_range": robust}


def texture_input(shader, role, root, cache, errors, uniform):
    inp = shader.GetInput(role)
    texture, channel = source(inp)
    where = f"{shader.GetPath()}.inputs:{role}"
    if not texture:
        literal = value(inp)
        if role in uniform and (not inp or not inp.GetAttr().GetConnections()):
            if role == "normal" and (literal is None or literal == Gf.Vec3f(0, 0, 1)):
                return {"normal": "geometry", "reason": uniform[role]}
            if (
                role == "diffuseColor"
                and isinstance(literal, Gf.Vec3f)
                and all(0 <= c <= 1 for c in literal)
            ):
                return {"constant": list(literal), "reason": uniform[role]}
        if role in ("metallic", "roughness") and isinstance(literal, (int, float)) and 0 <= literal <= 1:
            return {"constant": literal}
        if role == "metallic" and not inp:
            return {"constant": 0.0}
        errors.append(
            f"{where}: expected a connected UV texture"
            + (" or explicit roughness in [0,1]" if role == "roughness" else "")
        )
        return {}
    if texture.GetIdAttr().Get() != "UsdUVTexture":
        errors.append(f"{where}: expected UsdUVTexture in the universal preview network")
        return {}
    vector = role in ("diffuseColor", "normal")
    if channel not in (("rgb",) if vector else ("r", "g", "b", "a")):
        errors.append(f"{where}: invalid output channel {channel}")
    colorspace = value(texture.GetInput("sourceColorSpace"))
    # This portable bitmap profile expects conventional sRGB color and unsigned data maps.
    expected = "sRGB" if role == "diffuseColor" else "raw"
    if colorspace != expected:
        errors.append(f"{where}: use explicit {expected} for this bitmap (found {colorspace})")
    uv = uv_name(texture)
    if not isinstance(uv, str) or not uv:
        errors.append(f"{where}: connect st to a float2 primvar reader (optionally via UsdTransform2d)")
    if role == "normal":
        scale = value(texture.GetInput("scale"))
        bias = value(texture.GetInput("bias"))
        if (
            scale is None
            or bias is None
            or tuple(scale)[:3] != (2, 2, 2)
            or tuple(bias)[:3] != (-1, -1, -1)
        ):
            errors.append(f"{where}: decode unsigned OpenGL normals with RGB scale 2, bias -1")
    result = {"uv": uv, "channel": channel, "color_space": colorspace}
    file_input = texture.GetInput("file")
    asset = value(file_input)
    if not getattr(asset, "path", "") or not portable_path(asset.path):
        errors.append(f"{where}: supply a relative texture asset path")
        return result
    files = [asset.resolvedPath] if asset.resolvedPath else []
    if "<UDIM>" in asset.path:
        attr = file_input.GetValueProducingAttributes()[0]
        layer = attr.GetPropertyStack()[0].layer
        files = [p for p, _ in UsdShade.UdimUtils.ResolveUdimTilePaths(asset.path, layer)]
    if not files:
        errors.append(f"{where}: unresolved texture {asset.path}")
        return result
    try:
        result["images"] = [
            image_info(path, channel, root, cache, "rg" if role == "normal" else None)
            for path in files
        ]
        if role == "normal" and any(i["mode"] not in ("RGB", "RGBA") for i in result["images"]):
            errors.append(f"{where}: the tangent-space normal needs RGB components")
    except (OSError, ValueError) as exc:
        errors.append(f"{where}: {exc}")
    return result


def inspect_material(material, root, cache, errors, uniform):
    shader, _, _ = material.ComputeSurfaceSource()
    if not shader or shader.GetIdAttr().Get() != "UsdPreviewSurface":
        errors.append(
            f"{material.GetPath()}: author a universal UsdPreviewSurface terminal for offline review"
        )
        return {}
    result = {
        role: texture_input(shader, role, root, cache, errors, uniform)
        for role in ("diffuseColor", "roughness", "normal", "metallic")
    }
    relief = max(
        (image["p95_p05_range"] for role in ("normal", "roughness") for image in result[role].get("images", [])),
        default=0,
    )
    if relief < 12 and "normal" not in uniform:
        errors.append(
            f"{material.GetPath()}: normal and roughness maps have insufficient spatial variation "
            "(p95-p05 < 12/255); document a smooth-surface reason in [uniform]"
        )
    elif relief < 12:
        result["variation_override"] = uniform["normal"]
    if not any(image["varies"] for props in result.values() for image in props.get("images", [])):
        errors.append(
            f"{material.GetPath()}: the bound surface needs at least one spatially varied texture"
        )
    result["surface_outputs"] = [
        str(port.GetFullName()) for port in material.GetSurfaceOutputs() if port.HasConnectedSource()
    ]
    return result


def uv_check(prim, name, errors):
    where = f"{prim.GetPath()}.primvars:{name}"
    pv = UsdGeom.PrimvarsAPI(prim).FindPrimvarWithInheritance(name)
    coords = pv.ComputeFlattened() if pv else None
    if (
        coords is None
        or str(pv.GetTypeName()) not in ("texCoord2f[]", "float2[]", "texCoord2h[]", "texCoord2d[]")
        or not len(coords)
    ):
        errors.append(f"{where}: expected populated 2D UVs")
        return
    if not all(math.isfinite(component) for coord in coords for component in coord):
        errors.append(f"{where}: non-finite UVs")
        return
    origin = coords[0]
    direction = next((p - origin for p in coords if p != origin), None)
    if direction is None or not any(
        abs(direction[0] * (p[1] - origin[1]) - direction[1] * (p[0] - origin[0])) > 1e-14 for p in coords
    ):
        errors.append(f"{where}: UVs collapse to a point or a line")
    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        sizes = {
            "vertex": len(mesh.GetPointsAttr().Get() or []),
            "faceVarying": len(mesh.GetFaceVertexIndicesAttr().Get() or []),
            "uniform": len(mesh.GetFaceVertexCountsAttr().Get() or []),
            "constant": 1,
        }
        sizes["varying"] = sizes["vertex"]
        expected = sizes.get(pv.GetInterpolation())
        if len(coords) != expected:
            errors.append(
                f"{where}: {len(coords)} values for {pv.GetInterpolation()} UVs, expected {expected}"
            )


def bound_materials(prim, errors):
    binding = UsdShade.MaterialBindingAPI(prim)
    base, _ = binding.ComputeBoundMaterial(UsdShade.Tokens.preview)
    if not prim.IsA(UsdGeom.Mesh):
        if not base:
            errors.append(f"{prim.GetPath()}: rendered geometry has no bound material")
        return [base] if base else []
    mesh = UsdGeom.Mesh(prim)
    counts = mesh.GetFaceVertexCountsAttr().Get() or []
    indices = mesh.GetFaceVertexIndicesAttr().Get() or []
    points = mesh.GetPointsAttr().Get() or []
    valid, reason = mesh.ValidateTopology(indices, counts, len(points))
    if not valid or not counts or min(counts) < 3:
        errors.append(f"{prim.GetPath()}: invalid or empty mesh topology {reason}")
    covered = set()
    materials = []
    for subset in binding.GetMaterialBindSubsets():
        faces = list(subset.GetIndicesAttr().Get() or [])
        if (
            subset.GetElementTypeAttr().Get() != "face"
            or len(set(faces)) != len(faces)
            or any(f < 0 or f >= len(counts) or f in covered for f in faces)
        ):
            errors.append(f"{subset.GetPath()}: invalid or overlapping material subset")
        covered.update(faces)
        mat, _ = UsdShade.MaterialBindingAPI(subset).ComputeBoundMaterial(UsdShade.Tokens.preview)
        if mat and faces:
            materials.append(mat)
        elif faces:
            errors.append(f"{subset.GetPath()}: subset has no material")
    if len(covered) < len(counts):
        if base:
            materials.append(base)
        else:
            errors.append(f"{prim.GetPath()}: uncovered faces have no bound material")
    return materials
