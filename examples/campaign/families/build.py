"""Build a two-instance retail-package campaign from the existing YCB USD assets."""

from pathlib import Path

from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from harness.asset_family import create_family
from harness.author import initialize


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def surface_texture(path):
    # Deterministic fine paper fleck at 0.16 m repeat width, made for this example.
    image = Image.new("RGB", (64, 64))
    for y in range(64):
        for x in range(64):
            grain = ((x * 37 + y * 19 + x * y * 7) % 13) - 6
            image.putpixel((x, y), (174 + grain, 168 + grain, 153 + grain))
    image.save(path)


def build():
    source = REPO / "examples/campaign/ycb/models"
    family = HERE / "retail_package.usda"
    scene_path = HERE / "scene.usda"
    capture_dir = HERE / "capture"
    if not family.exists():
        create_family(
            [("cracker", source / "obj_000002.usdc"), ("sugar", source / "obj_000003.usdc")],
            family,
            default="cracker",
        )
    surface_texture(HERE / "paper_fleck.png")
    if scene_path.exists():
        stage = Usd.Stage.Open(str(scene_path))
        if not stage:
            raise ValueError(f"cannot open existing scene: {scene_path}")
        # The source scene is user owned after its first creation.
        if not capture_dir.exists():
            initialize(scene_path, capture_dir, [("retail_package", "/World/Packages/A"), ("retail_package", "/World/Packages/B")], width=640, height=480)
        return _append_edits(capture_dir / "campaign.toml")
    stage = Usd.Stage.CreateNew(str(scene_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    surface = UsdGeom.Mesh.Define(stage, "/World/Surface")
    surface.CreatePointsAttr([(-0.42, -0.3, 0), (0.42, -0.3, 0), (0.42, 0.3, 0), (-0.42, 0.3, 0)])
    surface.CreateFaceVertexCountsAttr([4])
    surface.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    surface.CreateSubdivisionSchemeAttr("none")
    uv = UsdGeom.PrimvarsAPI(surface).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    uv.Set([Gf.Vec2f(0, 0), Gf.Vec2f(5.25, 0), Gf.Vec2f(5.25, 3.75), Gf.Vec2f(0, 3.75)])
    material = UsdShade.Material.Define(stage, "/World/PaperMaterial")
    shader = UsdShade.Shader.Define(stage, "/World/PaperMaterial/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.88)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    reader = UsdShade.Shader.Define(stage, "/World/PaperMaterial/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    texture = UsdShade.Shader.Define(stage, "/World/PaperMaterial/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("paper_fleck.png"))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    for axis in ("wrapS", "wrapT"):
        texture.CreateInput(axis, Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    UsdShade.MaterialBindingAPI.Apply(surface.GetPrim()).Bind(material)
    for name, x in (("A", -0.13), ("B", 0.13)):
        package = UsdGeom.Xform.Define(stage, "/World/Packages/" + name)
        package.GetPrim().GetReferences().AddReference("retail_package.usda")
        package.AddTranslateOp().Set(Gf.Vec3d(x, 0, 0))
    stage.GetRootLayer().Save()
    if not capture_dir.exists():
        initialize(
            scene_path,
            capture_dir,
            [("retail_package", "/World/Packages/A"), ("retail_package", "/World/Packages/B")],
            width=640,
            height=480,
        )
    return _append_edits(capture_dir / "campaign.toml")


def _append_edits(config):
    body = config.read_text()
    if 'kind = "geometry_variant"' not in body:
        for name in ("A", "B"):
            body += (
                "\n[[edits]]\n"
                f'id = "package_{name.lower()}_geometry"\n'
                'kind = "geometry_variant"\n'
                f'prim = "/World/Packages/{name}"\n'
                'variant_set = "geometry"\n'
                'choices = ["cracker", "sugar"]\n'
            )
        config.write_text(body)
    return config


if __name__ == "__main__":
    print(build())
