"""Rebuild the bundled inspection scene with uv run --locked python scripts/build_example.py."""

from __future__ import annotations

import math
import random
import shutil
from datetime import date
from pathlib import Path

import zxingcpp
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

ROOT = Path(__file__).resolve().parents[1] / "examples" / "inspection"
TEX = ROOT / "textures"
TEX.mkdir(parents=True, exist_ok=True)


def authored_maps(name, rgb, rough, seed):
    rng = random.Random(seed)
    size = 512
    color = Image.new("RGB", (size, size))
    roughness = Image.new("L", (size, size))
    normal = Image.new("RGB", (size, size), (128, 128, 255))
    color_pixels, roughness_pixels, normal_pixels = color.load(), roughness.load(), normal.load()
    for y in range(size):
        for x in range(size):
            grain = rng.gauss(0, 4.0 if name == "carton" else 2.4)
            color_pixels[x, y] = tuple(max(0, min(255, round(k + grain))) for k in rgb)
            roughness_pixels[x, y] = max(0, min(255, round(rough * 255 + grain * 2.8)))
            amplitude = 12 if name == "carton" else (7 if name == "label" else 5)
            normal_pixels[x, y] = (
                max(0, min(255, round(128 + rng.gauss(0, amplitude)))),
                max(0, min(255, round(128 + rng.gauss(0, amplitude * 0.65)))),
                255,
            )
    if name == "carton":
        draw = ImageDraw.Draw(color)
        for _ in range(1700):
            x = rng.randrange(size)
            y = rng.randrange(size)
            length = rng.randrange(2, 8)
            tone = rng.randrange(-5, 6)
            fiber_color = tuple(max(0, min(255, k + tone)) for k in rgb)
            draw.line((x, y, min(size - 1, x + length), y), fill=fiber_color, width=1)
        # Fiber direction and cut-edge streaks remain legible at the detail camera.
        ndraw = ImageDraw.Draw(normal)
        for _ in range(850):
            x, y = rng.randrange(size), rng.randrange(size)
            ndraw.line((x, y, min(size - 1, x + rng.randrange(3, 18)), y), fill=(111, 136, 252))
    elif name == "tape":
        # Sparse transverse adhesive creases: mostly clear film, with a few
        # narrow changes in reflection rather than a printed brown strip.
        rdraw = ImageDraw.Draw(roughness)
        ndraw = ImageDraw.Draw(normal)
        for _ in range(95):
            x, y = rng.randrange(size), rng.randrange(size)
            length = rng.randrange(8, 65)
            end = min(size - 1, x + length)
            rdraw.line((x, y, end, y + rng.choice((-1, 0, 1))), fill=rng.randrange(83, 123))
            ndraw.line((x, y, end, y + 1), fill=(128, rng.choice((107, 148)), 253))
    color.save(TEX / f"{name}_base.png")
    roughness.save(TEX / f"{name}_rough.png")
    normal.save(TEX / f"{name}_normal.png")


authored_maps("carton", (161, 117, 72), 0.83, 103)
authored_maps("carton_wear", (133, 94, 57), 0.91, 104)
authored_maps("tape", (209, 175, 119), 0.36, 105)
authored_maps("polymer", (32, 41, 48), 0.38, 108)
authored_maps("label", (222, 218, 197), 0.72, 109)
authored_maps("wall_paint", (177, 184, 183), 0.79, 110)

label_size = (1300, 590)  # ten pixels per millimeter on the 130 x 59 mm physical label
label = Image.new("RGB", label_size, (236, 234, 224))
draw = ImageDraw.Draw(label)
font = ImageFont.load_default(size=60)
font_bold = ImageFont.load_default(size=76)
font_small = ImageFont.load_default(size=30)
ink = (24, 29, 32)
draw.text((60, 25), "ITEM  HST-042", font=font_bold, fill=ink)
draw.text((60, 112), "LOT  26-11-25-A", font=font, fill=ink)
draw.line((60, 177, 1240, 177), fill=(112, 114, 113), width=2)
digits = "0426112500014"
symbol = zxingcpp.create_barcode(digits, zxingcpp.BarcodeFormat.EAN13)
modules = memoryview(symbol.to_image(scale=1, add_quiet_zones=False))
module = 9
x0 = (label_size[0] - modules.shape[1] * module) // 2
for i in range(modules.shape[1]):
    if modules[0, i] == 0:
        guard = i < 3 or 45 <= i < 50 or i >= 92
        draw.rectangle((x0 + i * module, 209, x0 + (i + 1) * module - 1, 462 if guard else 440), fill=ink)
draw.text((label_size[0] // 2, 470), digits, anchor="ma", font=font_bold, fill=ink)
draw.text((60, 540), "VERIFY SEAM  /  ORIENT LABEL OUT", font=font_small, fill=(73, 79, 81))
label.save(TEX / "label_print_base.png")


# Satin brushed steel: elongated micro-grooves, not broad rust or stone mottling.
random.seed(111)
base = Image.new("RGB", (512, 512))
rough = Image.new("L", (512, 512))
normal = Image.new("RGB", (512, 512))
metal = Image.new("L", (512, 512), 255)
bp, rp, np = base.load(), rough.load(), normal.load()
for y in range(512):
    band = 2.5 * math.sin(y * 0.84) + 1.5 * math.sin(y * 2.1) + random.gauss(0, 1.1)
    for x in range(512):
        v = band + random.gauss(0, 0.9)
        bp[x, y] = (max(0, min(255, round(151 + v))),) * 3
        rp[x, y] = max(0, min(255, round(91 + v * 1.6)))
        np[x, y] = (128, max(0, min(255, round(128 + band * 1.8))), 255)
base.save(TEX / "brushed_steel_base.png")
rough.save(TEX / "brushed_steel_rough.png")
normal.save(TEX / "brushed_steel_normal.png")
metal.save(TEX / "brushed_steel_metal.png")

env = Usd.Stage.CreateNew(str(ROOT / "geometry.usdc"))
UsdGeom.SetStageUpAxis(env, UsdGeom.Tokens.z)
env.SetMetadata("metersPerUnit", 1.0)
world = UsdGeom.Xform.Define(env, "/World")
env.SetDefaultPrim(world.GetPrim())
UsdGeom.Scope.Define(env, "/World/Materials")
UsdGeom.Scope.Define(env, "/World/Environment")
props_group = UsdGeom.Scope.Define(env, "/World/Props")
Usd.ModelAPI(world.GetPrim()).SetKind("assembly")
Usd.ModelAPI(props_group.GetPrim()).SetKind("group")
UsdGeom.Scope.Define(env, "/World/Lights")
for fixture, z_offset in (("Tray", -0.0105), ("Handset", -0.037), ("Carton", -0.0065)):
    group = UsdGeom.Xform.Define(env, f"/World/Props/{fixture}")
    group.AddTranslateOp().Set(Gf.Vec3d(0, 0, z_offset))


def material(name, base, rough, normal, metallic=None, metal_value=0.0, base_scale=None, base_bias=None):
    path = f"/World/Materials/{name}"
    mat = UsdShade.Material.Define(env, path)
    surf = UsdShade.Shader.Define(env, path + "/PreviewSurface")
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    surf.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.5)
    surf.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(0.0)
    st = UsdShade.Shader.Define(env, path + "/UV")
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    st_out = st.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    def sample(kind, file, output, color_space, scale=None, bias=None):
        sh = UsdShade.Shader.Define(env, path + "/" + kind)
        sh.CreateIdAttr("UsdUVTexture")
        sh.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("./textures/" + file))
        sh.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
        sh.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        sh.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        sh.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
        if scale is not None:
            sh.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(scale)
        if bias is not None:
            sh.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(bias)
        return sh.CreateOutput(
            output, Sdf.ValueTypeNames.Float3 if output == "rgb" else Sdf.ValueTypeNames.Float
        )

    surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        sample("BaseColor", base, "rgb", "sRGB", base_scale, base_bias)
    )
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
        sample("Roughness", rough, "r", "raw")
    )
    surf.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
        sample("Normal", normal, "rgb", "raw", Gf.Vec4f(2, 2, 2, 1), Gf.Vec4f(-1, -1, -1, 0))
    )
    if metallic:
        surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(
            sample("Metallic", metallic, "r", "raw")
        )
    else:
        surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal_value)
    mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
    return mat


mats = {}
for name, asset in [
    ("Concrete", "concrete_floor_painted"),
    ("Wood", "plywood"),
    ("Steel", "metal_plate_02"),
]:
    mats[name] = material(
        name,
        f"{asset}_base.jpg",
        f"{asset}_rough.jpg",
        f"{asset}_normal.png",
        f"{asset}_metal.jpg" if name == "Steel" else None,
        base_scale=Gf.Vec4f(0.46, 0.46, 0.46, 1) if name == "Concrete" else None,
        base_bias=Gf.Vec4f(0.24, 0.24, 0.24, 0) if name == "Concrete" else None,
    )
for name in ("carton", "carton_wear", "tape", "polymer", "label", "wall_paint"):
    source = {"carton": "Cardboard002", "carton_wear": "Paper005"}.get(name)
    color_source = source or ("Cardboard002" if name == "tape" else None)
    mats[name.title()] = material(
        name.title(),
        # The clear film transmits the carton colour. Its own map carries
        # roughness and wrinkles, while the base colour follows the substrate.
        f"{color_source}_1K-JPG_Color.jpg" if color_source else f"{name}_base.png",
        f"{source}_1K-JPG_Roughness.jpg" if source else f"{name}_rough.png",
        f"{source}_1K-JPG_NormalGL.jpg" if source else f"{name}_normal.png",
    )
UsdShade.Shader.Get(env, "/World/Materials/Tape/PreviewSurface").CreateInput(
    "opacity", Sdf.ValueTypeNames.Float
).Set(0.38)
mats["LabelPrint"] = material("LabelPrint", "label_print_base.png", "label_rough.png", "label_normal.png")
mats["BrushedSteel"] = material(
    "BrushedSteel",
    "brushed_steel_base.png",
    "brushed_steel_rough.png",
    "brushed_steel_normal.png",
    "brushed_steel_metal.png",
)

# Bind the same portable material prim for rendering and contact. The friction
# values are design assumptions for a dry bench and can be varied per asset.
for name, static_friction, dynamic_friction in (
    ("Concrete", 0.8, 0.65), ("Wood", 0.62, 0.5),
    ("BrushedSteel", 0.48, 0.36), ("Carton", 0.72, 0.58),
    ("Polymer", 0.66, 0.51),
):
    physical = UsdPhysics.MaterialAPI.Apply(mats[name].GetPrim())
    physical.CreateStaticFrictionAttr(static_friction)
    physical.CreateDynamicFrictionAttr(dynamic_friction)
    physical.CreateRestitutionAttr(0.05)


def physical_material(prim, mat):
    prim.CreateRelationship("material:binding:physics").SetTargets([mat.GetPath()])


def rigid(path, mass, semantic):
    prim = env.GetPrimAtPath(path)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    prim.CreateAttribute("semantic:class", Sdf.ValueTypeNames.Token).Set(semantic)
    prim.CreateAttribute("semantic:instance", Sdf.ValueTypeNames.Token).Set(path.rsplit("/", 1)[-1].lower())
    prim.AddAppliedSchema("SemanticsLabelsAPI:class")
    prim.CreateAttribute("semantics:labels:class", Sdf.ValueTypeNames.TokenArray).Set([semantic])


# A bevelled cuboid mesh. Each face is subdivided at the bevel boundaries.
# UVs are authored per corner so real-world repeat sizes survive independent face orientation.
def box(path, center, size, mat, bevel=0.0, repeat=0.5, collision=False, rotate_uv=False):
    hx, hy, hz = (s / 2 for s in size)
    bevel = min(bevel, min(size) * 0.24)
    p = []
    uv = []
    counts = []
    indices = []
    normals = []
    frames = [
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
        ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
        ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
        ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
        ((0, 0, -1), (1, 0, 0), (0, -1, 0)),
    ]
    half = (hx, hy, hz)
    for outward, uaxis, vaxis in frames:
        ui = next(i for i, x in enumerate(uaxis) if x)
        vi = next(i for i, x in enumerate(vaxis) if x)
        umax = half[ui]
        vmax = half[vi]
        us = [-umax, -umax + bevel, umax - bevel, umax] if bevel else [-umax, umax]
        vs = [-vmax, -vmax + bevel, vmax - bevel, vmax] if bevel else [-vmax, vmax]
        grid = []
        for v in vs:
            row = []
            for u in us:
                q = [outward[i] * half[i] + uaxis[i] * u + vaxis[i] * v for i in range(3)]
                core = [max(-half[i] + bevel, min(half[i] - bevel, q[i])) for i in range(3)]
                d = [q[i] - core[i] for i in range(3)]
                norm = math.sqrt(sum(t * t for t in d))
                xyz = (
                    [center[i] + core[i] + bevel * d[i] / norm for i in range(3)]
                    if norm
                    else [center[i] + q[i] for i in range(3)]
                )
                row.append(len(p))
                p.append(Gf.Vec3f(*xyz))
                normals.append(Gf.Vec3f(*(t / norm for t in d)) if norm else Gf.Vec3f(*outward))
            grid.append(row)
        for j in range(len(vs) - 1):
            for i in range(len(us) - 1):
                ids = (grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i])
                counts.append(4)
                indices.extend(ids)
                for ii, jj in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                    u, v = (us[ii] + umax) / repeat, (vs[jj] + vmax) / repeat
                    uv.append(Gf.Vec2f(v, u) if rotate_uv else Gf.Vec2f(u, v))
    mesh = UsdGeom.Mesh.Define(env, path)
    # Weld shared geometric positions while retaining corner normals at each face vertex.
    corner_normals = [normals[i] for i in indices]
    unique_points = []
    point_lookup = {}
    welded_indices = []
    for i in indices:
        key = tuple(round(float(v), 8) for v in p[i])
        if key not in point_lookup:
            point_lookup[key] = len(unique_points)
            unique_points.append(p[i])
        welded_indices.append(point_lookup[key])
    mesh.CreatePointsAttr(unique_points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(welded_indices)
    mesh.CreateNormalsAttr(corner_normals)
    mesh.SetNormalsInterpolation("faceVarying")
    mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(unique_points))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(False)
    unique_uv = []
    uv_lookup = {}
    uv_indices = []
    for st in uv:
        key = (round(float(st[0]), 7), round(float(st[1]), 7))
        if key not in uv_lookup:
            uv_lookup[key] = len(unique_uv)
            unique_uv.append(st)
        uv_indices.append(uv_lookup[key])
    st_pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    st_pv.Set(unique_uv)
    st_pv.SetIndices(uv_indices)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
        physical_material(mesh.GetPrim(), mat)
    return mesh


def fluted_edge(path, x0, x1, y_center, z, mat):
    """A continuous thin corrugated paper ribbon between the liner skins."""
    points = []
    # FEFCO Table 1 B flute: 150 flutes/m, 2.4 mm peak-to-peak member height.
    # Eight samples per 6.67 mm cycle keep the sine visible in close inspection.
    count = 385
    for i in range(count):
        x = x0 + (x1 - x0) * i / (count - 1)
        y = y_center + 0.0012 * math.sin(2 * math.pi * (x - x0) * 150)
        for zz in (z, z + 0.0008):
            for yy in (y - 0.00010, y + 0.00010):
                points.append(Gf.Vec3f(x, yy, zz))
    faces = []
    for i in range(count - 1):
        a, b = 4 * i, 4 * (i + 1)
        faces.extend((
            ((a + 2, b + 2, b + 3, a + 3), "xy"),
            ((a, b, b + 2, a + 2), "xz"),
            ((a + 1, a + 3, b + 3, b + 1), "xz"),
            ((a, a + 1, b + 1, b), "xy"),
        ))
    end = 4 * (count - 1)
    faces.extend((((0, 2, 3, 1), "yz"), ((end, end + 1, end + 3, end + 2), "yz")))
    mesh = UsdGeom.Mesh.Define(env, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([i for face, _ in faces for i in face])
    corner_normals = []
    for face, _ in faces:
        normal = Gf.Cross(points[face[1]] - points[face[0]], points[face[2]] - points[face[0]]).GetNormalized()
        corner_normals.extend([normal] * 4)
    mesh.CreateNormalsAttr(corner_normals)
    mesh.SetNormalsInterpolation("faceVarying")
    mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(points))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(False)
    # Indexed face UVs follow the ribbon length and the exposed cross section.
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    uv_values, uv_indices, uv_lookup = [], [], {}
    for face, axes in faces:
        for index in face:
            point = points[index]
            a0, a1 = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[axes]
            origin0 = (x0, y_center, z)[a0]
            origin1 = (x0, y_center, z)[a1]
            uv = ((point[a0] - origin0) / 0.08, (point[a1] - origin1) / 0.08)
            key = tuple(round(v, 7) for v in uv)
            if key not in uv_lookup:
                uv_lookup[key] = len(uv_values)
                uv_values.append(Gf.Vec2f(*uv))
            uv_indices.append(uv_lookup[key])
    st.Set(uv_values)
    st.SetIndices(uv_indices)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    return mesh


def front_patch(path, outline, y, mat, repeat=0.12):
    """A small front-facing torn film or paper scuff; no rectangular backing quad."""
    def cross(a, b, c):
        return ((b[0] - a[0]) * (c[1] - a[1])
                - (b[1] - a[1]) * (c[0] - a[0]))

    # Ear clipping keeps every triangle inside a concave tear. A vertex fan
    # crosses the indented tape edge and makes a conspicuous missing wedge.
    signed_area = sum(
        outline[i][0] * outline[(i + 1) % len(outline)][1]
        - outline[(i + 1) % len(outline)][0] * outline[i][1]
        for i in range(len(outline))
    )
    remaining = list(range(len(outline)))
    if signed_area < 0:
        remaining.reverse()
    triangles = []
    while len(remaining) > 3:
        for pos, current in enumerate(remaining):
            prev = remaining[pos - 1]
            following = remaining[(pos + 1) % len(remaining)]
            a, b, c = outline[prev], outline[current], outline[following]
            if cross(a, b, c) <= 1e-12:
                continue
            if any(
                cross(a, b, outline[j]) >= -1e-12
                and cross(b, c, outline[j]) >= -1e-12
                and cross(c, a, outline[j]) >= -1e-12
                for j in remaining if j not in (prev, current, following)
            ):
                continue
            triangles.append((prev, current, following))
            remaining.pop(pos)
            break
        else:
            raise ValueError(f"Cannot triangulate front patch {path}")
    triangles.append(tuple(remaining))
    mesh = UsdGeom.Mesh.Define(env, path)
    points = [Gf.Vec3f(x, y + 0.00008 * math.sin(i * 2.3), z) for i, (x, z) in enumerate(outline)]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr([j for triangle in triangles for j in triangle])
    mesh.CreateNormalsAttr([Gf.Vec3f(0, -1, 0)] * len(points))
    mesh.SetNormalsInterpolation("vertex")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(points))
    uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    uv.Set([Gf.Vec2f(x / repeat, z / repeat) for x, z in outline])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    return mesh


def flap_cut_edge(path, x0, x1, y, z, mat):
    """B-flute visible on the front flap's XZ cut, between separate liner skins."""
    n = 385
    points = []
    for i in range(n):
        x = x0 + (x1 - x0) * i / (n - 1)
        wave = 0.00108 * math.sin(2 * math.pi * (x - x0) * 150)
        for zz in (z + wave - 0.00010, z + wave + 0.00010):
            points.append(Gf.Vec3f(x, y, zz))
    mesh = UsdGeom.Mesh.Define(env, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4] * (n - 1))
    mesh.CreateFaceVertexIndicesAttr([j for i in range(n - 1) for j in (2*i, 2*i+2, 2*i+3, 2*i+1)])
    mesh.CreateNormalsAttr([Gf.Vec3f(0, -1, 0)] * (2*n))
    mesh.SetNormalsInterpolation("vertex")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(points))
    uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    uv.Set([Gf.Vec2f((p[0]-x0)/0.08, (p[2]-z)/0.08) for p in points])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)


# Room and bench: real dimensions in meters.
box("/World/Environment/Floor", (0, 0, -0.06), (12, 12, 0.12), mats["Concrete"], 0.004, 2.0, True)
box("/World/Environment/BackWall", (0, 1.44, 1.15), (10, 0.08, 2.3), mats["Wall_Paint"], 0.004, 2.0)
for i in range(-5, 6):
    box(
        f"/World/Environment/WallPanelSeam_{i + 5}",
        (i * 0.8, 1.393, 1.33),
        (0.008, 0.002, 1.76),
        mats["Wall_Paint"],
        0.001,
        0.6,
    )
for x in (-0.72, 0.72):
    for y in (-0.36, 0.36):
        box(
            f"/World/Environment/BenchLeg_{'L' if x < 0 else 'R'}_{'F' if y < 0 else 'B'}",
            (x, y, 0.43),
            (0.055, 0.055, 0.86),
            mats["BrushedSteel"],
            0.003,
            0.6,
            True,
            True,
        )
box("/World/Environment/BenchFrame", (0, 0, 0.79), (1.55, 0.75, 0.055), mats["BrushedSteel"], 0.004, 0.75)
box("/World/Environment/Worktop", (0, 0, 0.87), (1.65, 0.85, 0.045), mats["Wood"], 0.006, 0.5, True)
# Dark antistatic inspection pad, its edge and corners are geometric.
box(
    "/World/Props/InspectionPad",
    (-0.38, -0.02, 0.8965),
    (0.58, 0.55, 0.008),
    mats["Polymer"],
    0.006,
    0.7,
    True,
)
# Metal parts tray and separate rolled rim / inset, exposing actual wall thickness.
box(
    "/World/Props/Tray/Base",
    (-0.38, -0.03, 0.918),
    (0.44, 0.34, 0.014),
    mats["BrushedSteel"],
    0.003,
    1.5,
    True,
)
for nm, c, s in [
    ("Front", (-0.38, -0.205, 0.938), (0.46, 0.012, 0.046)),
    ("Back", (-0.38, 0.145, 0.938), (0.46, 0.012, 0.046)),
    ("Left", (-0.61, -0.03, 0.938), (0.012, 0.35, 0.046)),
    ("Right", (-0.15, -0.03, 0.938), (0.012, 0.35, 0.046)),
]:
    box("/World/Props/Tray/Rim" + nm, c, s, mats["BrushedSteel"], 0.003, 1.5, True)
# Handset in tray: shell, inset display, side rails, camera island and lens rings.
box(
    "/World/Props/Handset/Body",
    (-0.38, -0.03, 0.956),
    (0.091, 0.174, 0.009),
    mats["Polymer"],
    0.004,
    0.25,
    True,
)
box(
    "/World/Props/Handset/Screen",
    (-0.38, -0.034, 0.9613),
    (0.082, 0.150, 0.0012),
    mats["Polymer"],
    0.001,
    0.25,
)
box(
    "/World/Props/Handset/DisplayInset",
    (-0.38, -0.034, 0.962),
    (0.072, 0.135, 0.0004),
    mats["BrushedSteel"],
    0.0001,
    1.2,
)
box(
    "/World/Props/Handset/CameraIsland",
    (-0.351, 0.042, 0.964),
    (0.025, 0.037, 0.003),
    mats["Polymer"],
    0.001,
    0.25,
)
for i, y in enumerate((0.033, 0.051)):
    box(
        f"/World/Props/Handset/Lens_{i}",
        (-0.351, y, 0.966),
        (0.012, 0.012, 0.0015),
        mats["BrushedSteel"],
        0.001,
        1.2,
    )
box(
    "/World/Props/Handset/SideButton",
    (-0.331, -0.041, 0.956),
    (0.002, 0.018, 0.002),
    mats["BrushedSteel"],
    0.0005,
    1.2,
)
# Open packing carton with base, true 2 mm walls, folded flaps, paper label.
box(
    "/World/Props/Carton/Base", (0.36, 0.025, 0.908), (0.32, 0.26, 0.018), mats["Carton"], 0.003, 0.45, True
)
for nm, c, s in [
    ("Front", (0.36, -0.106, 0.985), (0.32, 0.003, 0.146)),
    ("Back", (0.36, 0.156, 0.985), (0.32, 0.003, 0.146)),
    ("Left", (0.199, 0.025, 0.985), (0.003, 0.26, 0.146)),
    ("Right", (0.521, 0.025, 0.985), (0.003, 0.26, 0.146)),
]:
    box("/World/Props/Carton/Wall" + nm, c, s, mats["Carton"], 0.001, 0.45, True)
box(
    "/World/Props/Carton/FrontFlap", (0.36, -0.159, 1.05635), (0.31, 0.10, 0.00030), mats["Carton"], 0, 0.45
)
box("/World/Props/Carton/FrontFlapLowerLiner", (0.36, -0.159, 1.05365),
    (0.31, 0.10, 0.00030), mats["Carton"], 0, 0.45)
flap_cut_edge("/World/Props/Carton/FrontFlapFlute", 0.205, 0.515,
              -0.20915, 1.055, mats["Carton"])
box("/World/Props/Carton/BackFlap", (0.36, 0.210, 1.055), (0.31, 0.10, 0.003), mats["Carton"], 0.001, 0.45)
# The exposed 3 mm sheet shows two liner skins and a B-flute core.
for side, y in (("Front", -0.106), ("Back", 0.156)):
    fluted_edge(f"/World/Props/Carton/Corrugation{side}", 0.201, 0.519, y, 1.0580, mats["Carton"])
    for edge, direction in (("Outer", -1), ("Inner", 1)):
        box(
            f"/World/Props/Carton/CutLiner{side}{edge}",
            (0.36, y + direction * 0.00140, 1.05848),
            (0.318, 0.00020, 0.00096), mats["Carton"], 0, 0.08,
        )
front_patch("/World/Props/Carton/TapeSeam", [
    (0.350, 1.040), (0.370, 1.039), (0.371, 1.024), (0.369, 1.007),
    (0.370, 0.981), (0.368, 0.946), (0.350, 0.945), (0.349, 0.968),
    (0.351, 0.997), (0.349, 1.021),
], 0.1581, mats["Tape"])
front_patch("/World/Props/Carton/TapeVisible", [
    (0.477, 1.032), (0.496, 1.030), (0.498, 1.021), (0.495, 1.004),
    (0.497, 0.990), (0.495, 0.975), (0.498, 0.952), (0.496, 0.936),
    (0.478, 0.933), (0.476, 0.947), (0.478, 0.964), (0.475, 0.980),
    (0.477, 1.000), (0.475, 1.016),
], -0.10772, mats["Tape"])
# A ragged exposed-fibre patch, narrow enough to read as a lifted skin rather
# than a second label. Its nonuniform silhouette lies beyond the barcode ROI.
front_patch("/World/Props/Carton/EdgeWear", [
    (0.509, 0.938), (0.511, 0.937), (0.510, 0.934), (0.512, 0.932),
    (0.511, 0.929), (0.513, 0.926), (0.511, 0.923), (0.512, 0.920),
    (0.509, 0.917), (0.507, 0.918), (0.508, 0.922), (0.506, 0.924),
    (0.508, 0.927), (0.507, 0.930), (0.509, 0.933), (0.508, 0.936),
], -0.10780, mats["Carton_Wear"], 0.06)
box("/World/Props/Carton/Label", (0.36, -0.108, 0.987), (0.13, 0.0006, 0.059), mats["Label"], 0.0001, 0.25)
# Printed face, UVs map the entire graphic once over exactly 130 x 59 mm.
print_mesh = UsdGeom.Mesh.Define(env, "/World/Props/Carton/LabelPrint")
print_mesh.CreatePointsAttr(
    [
        Gf.Vec3f(0.295, -0.1085, 0.9575),
        Gf.Vec3f(0.425, -0.1085, 0.9575),
        Gf.Vec3f(0.425, -0.1085, 1.0165),
        Gf.Vec3f(0.295, -0.1085, 1.0165),
    ]
)
print_mesh.CreateFaceVertexCountsAttr([4])
print_mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
print_mesh.CreateNormalsAttr([Gf.Vec3f(0, -1, 0)] * 4)
print_mesh.SetNormalsInterpolation("vertex")
print_mesh.CreateSubdivisionSchemeAttr("none")
print_mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(print_mesh.GetPointsAttr().Get()))
print_st = UsdGeom.PrimvarsAPI(print_mesh).CreatePrimvar(
    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
)
print_st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])
print_st.SetIndices([0, 1, 2, 3])
UsdShade.MaterialBindingAPI.Apply(print_mesh.GetPrim()).Bind(mats["LabelPrint"])
box("/World/Props/WearCoupon", (0.07, 0.28, 0.8935), (0.12, 0.08, 0.002), mats["Steel"], 0.0005, 2.0)
# Bench detail: steel calibration ruler and spacing grooves as individual relief.
box("/World/Props/Ruler", (0.07, -0.24, 0.8935), (0.33, 0.028, 0.002), mats["BrushedSteel"], 0.0005, 1.5)
for i in range(13):
    box(
        f"/World/Props/RulerTick_{i}",
        (-0.083 + i * 0.025, -0.24, 0.8948),
        (0.0014, 0.011 if i % 5 else 0.018, 0.0007),
        mats["Label"],
        0,
        0.25,
    )

rigid("/World/Props/Tray", 0.43, "parts_tray")
rigid("/World/Props/Handset", 0.24, "handheld_scanner")
rigid("/World/Props/Carton", 0.18, "carton")


# USD Lux intensity is authored in renderer scene units, with exposure = 0.
# A measured 3.5-stop rig gain corrects underexposure in the fixed L4 review.
# These are studio lookdev settings, not measured illuminance at a work site.
LIGHT_GAIN = 2 ** 3.5
def light(name, kind, xyz, intensity, color, width=None, height=None):
    p = f"/World/Lights/{name}"
    l = UsdLux.RectLight.Define(env, p) if kind == "rect" else UsdLux.SphereLight.Define(env, p)
    l.CreateIntensityAttr(intensity * LIGHT_GAIN)
    l.CreateColorAttr(Gf.Vec3f(*color))
    if kind == "rect":
        l.CreateWidthAttr(width)
        l.CreateHeightAttr(height)
    else:
        l.CreateRadiusAttr(0.16)
    l.AddTranslateOp().Set(Gf.Vec3d(*xyz))
    if kind == "rect":
        l.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 0))  # lights point down -Z
    return l


light("OverheadSoftbox", "rect", (-0.4, -0.1, 2.55), 15.2, (1, 0.96, 0.91), 1.1, 0.75)
light("SteelEdge", "rect", (0.95, 0.35, 2.05), 10.8, (0.79, 0.88, 1), 0.35, 1.1)
light("FrontFill", "sphere", (0, -1.1, 1.75), 3.8, (1, 0.93, 0.84))
front = UsdLux.RectLight.Define(env, "/World/Lights/LabelSoftbox")
front.CreateIntensityAttr(25.0 * LIGHT_GAIN)
front.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))
front.CreateWidthAttr(0.72)
front.CreateHeightAttr(0.46)
front_view = Gf.Matrix4d().SetLookAt(
    Gf.Vec3d(0.36, -0.95, 1.42), Gf.Vec3d(0.36, -0.10, 1.0), Gf.Vec3d(0, 0, 1)
)
front.AddTransformOp().Set(front_view.GetInverse())
dome = UsdLux.DomeLight.Define(env, "/World/Lights/Ambient")
dome.CreateIntensityAttr(0.18 * LIGHT_GAIN)
dome.CreateColorAttr(Gf.Vec3f(0.89, 0.93, 1.0))
lighting = Usd.Stage.CreateNew(str(ROOT / "lighting.usda"))
UsdGeom.SetStageUpAxis(lighting, UsdGeom.Tokens.z)
lighting.SetMetadata("metersPerUnit", 1.0)
UsdGeom.Xform.Define(lighting, "/World")
Sdf.CopySpec(env.GetRootLayer(), "/World/Lights", lighting.GetRootLayer(), "/World/Lights")
lighting.GetRootLayer().Save()
hdri = UsdLux.DomeLight.Get(lighting, "/World/Lights/Ambient")
hdri.CreateTextureFileAttr(Sdf.AssetPath("./lighting/empty_warehouse_01_1k.hdr"))
hdri.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
lighting.GetRootLayer().Save()
env.RemovePrim("/World/Lights")
env.GetRootLayer().Save()
# Thin stable human-readable environment entry; heavy geometry lives in binary USD.
env_entry = Usd.Stage.CreateNew(str(ROOT / "environment.usda"))
env_entry.GetRootLayer().subLayerPaths = ["./geometry.usdc"]
env_entry.SetMetadata("defaultPrim", "World")
env_entry.SetMetadata("metersPerUnit", 1.0)
UsdGeom.SetStageUpAxis(env_entry, UsdGeom.Tokens.z)
env_entry.GetRootLayer().Save()

camstage = Usd.Stage.CreateNew(str(ROOT / "cameras.usda"))
UsdGeom.SetStageUpAxis(camstage, UsdGeom.Tokens.z)
camstage.SetMetadata("metersPerUnit", 1.0)
UsdGeom.Xform.Define(camstage, "/World/Cameras")


def camera(name, eye, target, focal):
    cam = UsdGeom.Camera.Define(camstage, f"/World/Cameras/{name}")
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    cam.AddTransformOp().Set(view.GetInverse())
    cam.CreateFocalLengthAttr(focal / 100)
    cam.CreateHorizontalApertureAttr(0.36)
    cam.CreateVerticalApertureAttr(0.24)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 20))
    cam.CreateFocusDistanceAttr(math.dist(eye, target))
    cam.CreateFStopAttr(8)
    return cam


camera("Overview", (1.75, -2.32, 1.64), (0, 0, 0.35), 42)
camera("Detail", (0.51, -0.72, 1.19), (0.36, -0.07, 1.00), 58)
camstage.GetRootLayer().Save()

rootstage = Usd.Stage.CreateNew(str(ROOT / "scene.usda"))
rootstage.GetRootLayer().subLayerPaths = ["environment.usda", "lighting.usda", "cameras.usda"]
rootstage.SetMetadata("defaultPrim", "World")
rootstage.SetMetadata("metersPerUnit", 1.0)
UsdGeom.SetStageUpAxis(rootstage, UsdGeom.Tokens.z)
UsdPhysics.Scene.Define(rootstage, "/World/PhysicsScene")
rootstage.GetRootLayer().Save()
for name, required_materials in (
    ("Carton", ("Carton", "Carton_Wear", "Tape", "Label", "LabelPrint")),
    ("Handset", ("Polymer", "BrushedSteel")),
    ("Tray", ("BrushedSteel",)),
):
    local_origin = {
        "Carton": Gf.Vec3f(0.36, 0.025, 0.899),
        "Handset": Gf.Vec3f(-0.38, -0.03, 0.9515),
        "Tray": Gf.Vec3f(-0.38, -0.03, 0.911),
    }[name]
    package = ROOT / "assets" / name.lower()
    (package / "textures").mkdir(parents=True, exist_ok=True)
    asset = Usd.Stage.CreateNew(str(package / f"{name.lower()}.usdc"))
    UsdGeom.SetStageUpAxis(asset, UsdGeom.Tokens.z)
    asset.SetMetadata("metersPerUnit", 1.0)
    asset_root = UsdGeom.Xform.Define(asset, f"/{name}")
    asset.SetDefaultPrim(asset_root.GetPrim())
    Sdf.CopySpec(env.GetRootLayer(), f"/World/Props/{name}", asset.GetRootLayer(), f"/{name}")
    for prim in Usd.PrimRange(asset_root.GetPrim()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        shifted = [Gf.Vec3f(p - local_origin) for p in mesh.GetPointsAttr().Get()]
        mesh.GetPointsAttr().Set(shifted)
        mesh.GetExtentAttr().Set(UsdGeom.PointBased.ComputeExtent(shifted))
    root_translate = asset_root.GetPrim().GetAttribute("xformOp:translate")
    if root_translate:
        root_translate.Set(Gf.Vec3d(0, 0, 0))
    Usd.ModelAPI(asset_root.GetPrim()).SetKind("component")
    asset_root.GetPrim().CreateAttribute(
        "simready:assetFrame:forwardDirection", Sdf.ValueTypeNames.Vector3f,
        custom=True, variability=Sdf.VariabilityUniform,
    ).Set(Gf.Vec3f(0, -1, 0))
    UsdGeom.Scope.Define(asset, f"/{name}/Materials")
    for material_name in required_materials:
        Sdf.CopySpec(
            env.GetRootLayer(), f"/World/Materials/{material_name}",
            asset.GetRootLayer(), f"/{name}/Materials/{material_name}",
        )
        material_prim = asset.GetPrimAtPath(f"/{name}/Materials/{material_name}")
        material_prim.AddAppliedSchema("SemanticsLabelsAPI:material")
        material_prim.CreateAttribute(
            "semantics:labels:material", Sdf.ValueTypeNames.TokenArray
        ).Set([{
            "Carton": "cardboard", "Carton_Wear": "cardboard", "Tape": "adhesive_tape",
            "Label": "paper", "LabelPrint": "paper",
            "Polymer": "plastic", "BrushedSteel": "steel",
        }[material_name]])
    for prim in Usd.PrimRange(asset_root.GetPrim()):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        binding = prim.GetRelationship("material:binding")
        targets = binding.GetTargets() if binding else []
        if not targets:
            continue
        remapped = UsdShade.Material.Get(asset, f"/{name}/Materials/{targets[0].name}")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(remapped)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            physical_material(prim, remapped)
    for prim in Usd.PrimRange(asset_root.GetPrim()):
        if not prim.IsA(UsdShade.Shader):
            continue
        file_input = UsdShade.Shader(prim).GetInput("file")
        if file_input:
            source_file = Path(file_input.Get().path).name
            shutil.copy2(TEX / source_file, package / "textures" / source_file)
            file_input.Set(Sdf.AssetPath("./textures/" + source_file))
    if name in ("Carton", "Handset"):
        variant = asset_root.GetPrim().GetVariantSets().AddVariantSet("surface_finish")
        material_name = "Carton" if name == "Carton" else "Polymer"
        shader_path = f"/{name}/Materials/{material_name}/Roughness"
        for selection, scale, bias in (
            ("standard", 1.0, 0.0),
            ("worn" if name == "Carton" else "matte", 0.85 if name == "Carton" else 1.1, 0.12 if name == "Carton" else 0.08),
        ):
            variant.AddVariant(selection)
            variant.SetVariantSelection(selection)
            with variant.GetVariantEditContext():
                shader = UsdShade.Shader.Get(asset, shader_path)
                shader.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(scale, scale, scale, 1))
                shader.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(bias, bias, bias, 0))
        variant.SetVariantSelection("standard")
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(asset_root.GetPrim()).ComputeAlignedRange()
    size = bbox.GetSize()
    asset.GetRootLayer().customLayerData = {
        "SimReady_Metadata": {
            "author": "Isaac Harness example generator",
            "asset_name": name,
            "asset_type": "prop",
            "category": {"Carton": "packaging", "Handset": "handheld device", "Tray": "parts tray"}[name],
            "source_file": "scripts/build_example.py",
            "usd_date_generated": date.today().isoformat(),
            "rigid_body_count": 1,
            "asset_extents": Gf.Vec3d(*[float(size[i]) for i in range(3)]),
            "mass": {"Carton": 0.18, "Handset": 0.24, "Tray": 0.43}[name],
            "qcode": {"Carton": "Q1395006", "Handset": "Q807980", "Tray": "Q613972"}[name],
            "asset_license": "NOASSERTION",
        }
    }
    asset.GetRootLayer().Save()
    env.RemovePrim(f"/World/Props/{name}")
    placed = UsdGeom.Xform.Define(env, f"/World/Props/{name}")
    placed.GetPrim().GetReferences().AddReference(
        f"./assets/{name.lower()}/{name.lower()}.usdc"
    )
    placed.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(
        local_origin[0], local_origin[1],
        local_origin[2] + {"Tray": -0.0105, "Handset": -0.037, "Carton": -0.0065}[name],
    ))
env.GetRootLayer().Save()
print(f"Built {ROOT / 'scene.usda'}")
