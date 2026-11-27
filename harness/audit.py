"""Evaluate the composed, default-time scene under each configured lighting case."""

import math
from pathlib import Path

from pxr import Gf, Tf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdUtils

from .materials import bound_materials, inspect_material, portable_path, uv_check
from .project import file_hash


def is_rendered(prim):
    imageable = UsdGeom.Imageable(prim)
    return (
        imageable
        and imageable.ComputeVisibility() != "invisible"
        and imageable.ComputePurpose() in ("default", "render")
    )


def dependencies(root, stage_path, hashes, errors):
    layers, assets, missing = UsdUtils.ComputeAllDependencies(str(stage_path))
    errors.extend(f"unresolved asset: {name}" for name in missing)
    for layer in layers:
        for group in UsdUtils.ExtractExternalReferences(layer.realPath):
            for path in group:
                if not portable_path(path):
                    errors.append(f"{layer.GetDisplayName()}: non-portable asset reference {path}")
    files = {}
    for path in sorted(set([layer.realPath for layer in layers] + list(assets))):
        file = Path(path).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            errors.append(f"dependency outside or missing from scene package: {path}")
            continue
        if file not in hashes:
            hashes[file] = file_hash(file)
        files[file.relative_to(root).as_posix()] = hashes[file]
    return files


def inspect_view(stage, view, width, bounds, errors):
    camera = UsdGeom.Camera.Get(stage, view.camera)
    if not camera or not camera.GetPrim().IsActive():
        errors.append(f"{view.camera}: expected an active Camera")
        return {}
    time = Usd.TimeCode.Default()
    optics = camera.GetCamera(time)
    frustum = optics.frustum
    mm = UsdGeom.GetStageMetersPerUnit(stage) * 100
    aperture = [optics.horizontalAperture * mm, optics.verticalAperture * mm]
    near, far = camera.GetClippingRangeAttr().Get()
    if (
        not all(math.isfinite(v) and v > 0 for v in (*aperture, optics.focalLength, near, far))
        or far <= near
    ):
        errors.append(f"{view.camera}: invalid lens, filmback or clipping range")
        return {}
    result = {
        "camera": view.camera,
        "target": view.target,
        "inspect": view.inspect,
        "reference_size": [width, max(1, round(width * aperture[1] / aperture[0]))],
        "filmback_mm": aperture,
        "focal_length_mm": optics.focalLength * mm,
        "f_stop": optics.fStop,
        "focus_distance_m": optics.focusDistance * mm / 100,
        "camera_to_world": [list(row) for row in optics.transform],
    }
    if view.focal_length_mm is not None and not math.isclose(
        result["focal_length_mm"], view.focal_length_mm, rel_tol=1e-4
    ):
        errors.append(
            f"{view.camera}: expected {view.focal_length_mm} mm lens; found {result['focal_length_mm']:.2f} mm"
        )
    for path in view.include:
        prim = stage.GetPrimAtPath(path)
        if not prim or not is_rendered(prim):
            errors.append(f"{view.camera}: required subject {path} is missing or hidden")
            continue
        bound = bounds.ComputeWorldBound(prim)
        if bound.GetRange().IsEmpty() or not all(
            frustum.Intersects(bound.GetMatrix().Transform(bound.GetRange().GetCorner(i))) for i in range(8)
        ):
            errors.append(f"{view.camera}: required subject {path} is cropped")
    result["required_in_frame"] = list(view.include)
    target = stage.GetPrimAtPath(view.target)
    if not target or not target.IsA(UsdGeom.Imageable) or not is_rendered(target):
        errors.append(f"{view.target}: missing or hidden inspection target")
        return result
    bbox = bounds.ComputeWorldBound(target)
    box = bbox.GetRange()
    if box.IsEmpty() or not frustum.Intersects(bbox):
        errors.append(f"{view.camera}: target {view.target} is outside the camera or has empty bounds")
        return result
    # Project the tight bound in its own coordinate system; do not project a world-aligned re-boxing.
    view_matrix = frustum.ComputeViewMatrix()
    projection = frustum.ComputeProjectionMatrix()
    corners = [view_matrix.Transform(bbox.GetMatrix().Transform(box.GetCorner(i))) for i in range(8)]
    # Clipping the 12 box edges at near/far also handles a box straddling the camera.
    points = [p for p in corners if near <= -p[2] <= far]
    for i, a in enumerate(corners):
        for axis in (1, 2, 4):
            if i & axis:
                continue
            b = corners[i ^ axis]
            for z in (-near, -far):
                if min(a[2], b[2]) < z < max(a[2], b[2]):
                    points.append(a + (b - a) * ((z - a[2]) / (b[2] - a[2])))
    projected = [projection.Transform(Gf.Vec3d(p)) for p in points]
    fraction = 0.0
    if projected:
        xs = [(p[0] + 1) / 2 for p in projected]
        ys = [(1 - p[1]) / 2 for p in projected]  # image origin is top left
        left, right = max(0, min(xs)), min(1, max(xs))
        top, bottom = max(0, min(ys)), min(1, max(ys))
        if right > left and bottom > top:
            result["target_rect"] = [left, top, right, bottom]
            fraction = right - left
    result["projected_width_px"] = round(width * fraction, 2)
    if width * fraction < view.min_width_px:
        errors.append(
            f"{view.camera}: {view.target} spans {width * fraction:.1f}px, needs {view.min_width_px}px"
        )
    return result


def inspect_support(stage, bounds, obj_path, surface_path, tolerance, errors):
    ranges = []
    for path in (obj_path, surface_path):
        prim = stage.GetPrimAtPath(path)
        if not prim or not is_rendered(prim):
            errors.append(f"{path}: missing or hidden support check geometry")
            return {}
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            errors.append(f"{path}: support geometry needs an authored collider")
        extent = bounds.ComputeWorldBound(prim).ComputeAlignedRange()
        if extent.IsEmpty():
            errors.append(f"{path}: missing support check bounds")
            return {}
        ranges.append(extent)
    obj, surface = ranges
    overlap = all(min(obj.max[i], surface.max[i]) > max(obj.min[i], surface.min[i]) for i in (0, 1))
    gap = (obj.min[2] - surface.max[2]) * UsdGeom.GetStageMetersPerUnit(stage) * 1000
    if not overlap or abs(gap) > tolerance:
        errors.append(
            f"{obj_path} on {surface_path}: XY overlap={overlap}, vertical gap {gap:.3f} mm (±{tolerance} mm)"
        )
    return {
        "object": obj_path,
        "surface": surface_path,
        "gap_mm": round(gap, 4),
        "xy_overlap": overlap,
        "tolerance_mm": tolerance,
    }


def inspect_rigid_bodies(stage, errors):
    bodies = {}
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        path = str(prim.GetPath())
        mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get() if prim.HasAPI(UsdPhysics.MassAPI) else None
        colliders = [str(child.GetPath()) for child in Usd.PrimRange(prim) if child.HasAPI(UsdPhysics.CollisionAPI)]
        if mass is None or not math.isfinite(mass) or mass <= 0:
            errors.append(f"{path}: rigid body needs a finite positive mass")
        if not colliders:
            errors.append(f"{path}: rigid body needs descendant colliders")
        for child in Usd.PrimRange(prim):
            if child.GetName().startswith(("Wall", "Rim")) and child.IsA(UsdGeom.Gprim) and not child.HasAPI(UsdPhysics.CollisionAPI):
                errors.append(f"{child.GetPath()}: side wall needs an authored collider")
        bodies[path] = {"mass_kg": mass, "colliders": colliders}
    return bodies


def audit_case(project, path, image_cache, hashes):
    errors = []
    result = {
        "stage": path.relative_to(project.root).as_posix(),
        "errors": errors,
        "visible_gprims": 0,
        "materials": {},
        "lights": {},
        "views": {},
    }
    try:
        stage = Usd.Stage.Open(str(path))
        if not stage:
            raise ValueError(f"Could not open {path}")
        errors.extend(str(issue) for issue in stage.GetCompositionErrors())
        result["dependencies_sha256"] = dependencies(project.root, path, hashes, errors)
        result["up_axis"] = UsdGeom.GetStageUpAxis(stage)
        result["meters_per_unit"] = UsdGeom.GetStageMetersPerUnit(stage)
        if not stage.GetDefaultPrim():
            errors.append("declare a valid defaultPrim for referencing the environment")
        if not stage.HasAuthoredMetadata("upAxis") or result["up_axis"] != "Z":
            errors.append("declare upAxis = Z for this Isaac environment profile")
        if not stage.HasAuthoredMetadata("metersPerUnit") or result["meters_per_unit"] != 1:
            errors.append("declare metersPerUnit = 1 and author dimensions in meters")

        materials = result["materials"]
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            if not is_rendered(prim):
                continue
            if prim.HasAPI(UsdLux.LightAPI):
                light = UsdLux.LightAPI(prim)
                intensity = light.GetIntensityAttr().Get()
                exposure = light.GetExposureAttr().Get()
                color = light.GetColorAttr().Get()
                if (
                    intensity is not None
                    and intensity > 0
                    and math.isfinite(intensity)
                    and math.isfinite(exposure)
                    and max(color) > 0
                ):
                    result["lights"][str(prim.GetPath())] = {
                        "type": prim.GetTypeName(),
                        "intensity": intensity,
                        "exposure": exposure,
                    }
            if not prim.IsA(UsdGeom.Gprim):
                continue
            result["visible_gprims"] += 1
            uv_names = set()
            for material in bound_materials(prim, errors):
                key = str(material.GetPath())
                if key not in materials:
                    materials[key] = inspect_material(
                        material, project.root, image_cache, errors, project.uniform.get(key, {})
                    )
                for role in ("diffuseColor", "normal", "roughness", "metallic"):
                    uv = materials[key].get(role, {}).get("uv")
                    if uv:
                        uv_names.add(uv)
            for name in sorted(uv_names):
                uv_check(prim, name, errors)
        if not result["visible_gprims"]:
            errors.append("stage contains no visible render geometry")
        if not result["lights"]:
            errors.append("stage contains no visible light with positive authored contribution")
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=False)
        for name, view in project.views.items():
            result["views"][name] = inspect_view(stage, view, project.width, bounds, errors)
        result["supports"] = [
            inspect_support(stage, bounds, obj, surface, tol, errors)
            for obj, surface, tol in project.supports
        ]
        result["rigid_bodies"] = inspect_rigid_bodies(stage, errors)
    except (Tf.ErrorException, OSError, ValueError) as exc:
        errors.append(str(exc).strip())
    result["status"] = "fail" if errors else "pass"
    return result


def audit(project, selected=None):
    names = selected or list(project.cases)
    for name in names:
        if name not in project.cases:
            raise ValueError(f"Unknown case {name!r}. Available: {', '.join(project.cases)}")
    image_cache, hashes = {}, {}
    cases = {name: audit_case(project, project.cases[name], image_cache, hashes) for name in names}
    return {
        "project": project.name,
        "status": "fail" if any(c["errors"] for c in cases.values()) else "pass",
        "usd_version": ".".join(map(str, Usd.GetVersion())),
        "profile_sha256": file_hash(project.profile),
        "cases": cases,
    }
