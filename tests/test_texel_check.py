from pathlib import Path

import pytest

from harness.texel_check import check, main


SOURCE = Path(__file__).parents[1] / "examples/campaign/families/scene.usda"


def fixture_scene(tmp_path, edits=()):
    scene = SOURCE.read_text()
    for old, new in edits:
        assert old in scene
        scene = scene.replace(old, new)
    (tmp_path / "paper_fleck.png").write_bytes((SOURCE.parent / "paper_fleck.png").read_bytes())
    output = tmp_path / "scene.usda"
    output.write_text(scene)
    return output


def test_corrected_family_and_threshold(tmp_path):
    report = check(SOURCE, ["/World/Surface"], 390)
    assert report["ok"]
    item = report["meshes"][0]["materials"][0]
    assert item["repeat_span"] == pytest.approx([5.25, 3.75])
    assert item["min_texels_per_metre"] == pytest.approx(400, rel=1e-5)
    assert len(item["sha256"]) == 64
    assert not check(SOURCE, ["/World/Surface"], 410)["ok"]


def test_default_wrap_fails_even_with_high_density(tmp_path):
    scene = fixture_scene(tmp_path, [
        ('            token inputs:wrapS = "repeat"\n', ""),
        ('            token inputs:wrapT = "repeat"\n', ""),
    ])
    out = tmp_path / "report.json"
    assert main([str(scene), "--mesh", "/World/Surface", "--out", str(out)]) == 1
    text = out.read_text()
    assert "wrapS must be explicitly repeat or mirror" in text
    assert "wrapT must be explicitly repeat or mirror" in text


def test_world_transform_and_meters_per_unit(tmp_path):
    scene = fixture_scene(tmp_path, [
        ('    def Mesh "Surface" (',
         '    double3 xformOp:scale = (2, 2, 2)\n'
         '    uniform token[] xformOpOrder = ["xformOp:scale"]\n'
         '    def Mesh "Surface" ('),
    ])
    report = check(scene, ["/World/Surface"])
    assert report["ok"]
    density = report["meshes"][0]["materials"][0]["min_texels_per_metre"]
    assert density == pytest.approx(200, rel=1e-5)
    scene.write_text(scene.read_text().replace("metersPerUnit = 1", "metersPerUnit = 0.5"))
    report = check(scene, ["/World/Surface"])
    density = report["meshes"][0]["materials"][0]["min_texels_per_metre"]
    assert density == pytest.approx(400, rel=1e-5)


def test_indexed_face_varying_and_unsupported_st(tmp_path):
    scene = fixture_scene(tmp_path, [
        ('            interpolation = "vertex"\n        )',
         '            interpolation = "faceVarying"\n        )\n'
         '        int[] primvars:st:indices = [0, 1, 2, 3]'),
    ])
    assert check(scene, ["/World/Surface"])["ok"]
    # Keep a separate file in the same package so the texture still resolves.
    broken = scene.read_text().replace(
        'inputs:st.connect = </World/PaperMaterial/UV.outputs:result>',
        'inputs:st = (0, 0)',
    )
    (tmp_path / "unsupported.usda").write_text(broken)
    report = check(tmp_path / "unsupported.usda", ["/World/Surface"])
    assert not report["ok"]
    assert "st must connect directly" in report["errors"][0]
