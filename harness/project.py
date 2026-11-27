"""Read scene profiles and identify the inputs used for a review."""

import hashlib
import json
import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from pxr import Sdf


@dataclass(frozen=True)
class View:
    camera: str
    target: str
    min_width_px: int
    inspect: str
    reference: Path | None
    focal_length_mm: float | None
    expected_barcode: str | None
    include: tuple[str, ...]


@dataclass(frozen=True)
class Project:
    name: str
    root: Path
    profile: Path
    width: int
    cases: dict[str, Path]
    views: dict[str, View]
    supports: list[tuple[str, str, float]]
    uniform: dict[str, dict[str, str]]


def inside(root, value):
    if not isinstance(value, str) or not value.strip() or Path(value).is_absolute():
        raise ValueError(f"Use a relative project path: {value!r}")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path leaves the project directory: {value}")
    return path


def prim_path(value):
    if not isinstance(value, str) or not Sdf.Path.IsValidPathString(value):
        raise ValueError(f"Invalid USD prim path: {value!r}")
    path = Sdf.Path(value)
    if not path.IsAbsolutePath() or not path.IsPrimPath():
        raise ValueError(f"Expected an absolute USD prim path: {value!r}")
    return value


def positive_int(value, field):
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def load_project(path):
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "scene.toml"
    with path.open("rb") as stream:
        cfg = tomllib.load(stream)
    root = path.parent
    cases = cfg.get("cases")
    views = cfg.get("views")
    if not isinstance(cases, dict) or not cases or not isinstance(views, dict) or not views:
        raise ValueError("scene.toml requires [cases] and [views.<name>] tables")
    for name in (*cases, *views):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError(f"Invalid case or view name: {name!r}")
    parsed = {}
    for name, values in views.items():
        if not isinstance(values, dict):
            raise TypeError(f"Expected a table for views.{name}")
        brief = values.get("inspect")
        if not isinstance(brief, str) or not brief.strip():
            raise ValueError(f"Describe the visual review in views.{name}.inspect")
        reference = inside(root, values["reference"]) if "reference" in values else None
        if reference and not reference.is_file():
            raise ValueError(f"Missing reference photograph: {reference}")
        focal = values.get("focal_length_mm")
        if focal is not None and (
            type(focal) not in (int, float) or not math.isfinite(focal) or focal <= 0
        ):
            raise ValueError(f"views.{name}.focal_length_mm must be positive and finite")
        barcode = values.get("expected_barcode")
        if barcode is not None and (not isinstance(barcode, str) or not barcode):
            raise ValueError(f"views.{name}.expected_barcode must be a nonempty string")
        include = values.get("include", [])
        if not isinstance(include, list):
            raise TypeError(f"views.{name}.include must be a list of prim paths")
        parsed[name] = View(
            prim_path(values.get("camera")),
            prim_path(values.get("target")),
            positive_int(values.get("min_width_px", 1), "min_width_px"),
            brief,
            reference,
            focal,
            barcode,
            tuple(prim_path(path) for path in include),
        )
    supports = []
    for pair in cfg.get("supports", []):
        if not isinstance(pair, dict):
            raise TypeError("Use [[supports]] tables for horizontal contacts")
        tolerance = pair.get("tolerance_mm", 1.0)
        if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance_mm must be a finite non-negative length")
        supports.append((prim_path(pair.get("object")), prim_path(pair.get("surface")), tolerance))
    uniform = cfg.get("uniform", {})
    if not isinstance(uniform, dict):
        raise TypeError('Use [uniform."/Material/Path"] to document uniform inputs')
    for path_name, fields in uniform.items():
        prim_path(path_name)
        if (
            not isinstance(fields, dict)
            or not fields
            or any(
                k not in {"diffuseColor", "normal"} or not isinstance(v, str) or not v.strip()
                for k, v in fields.items()
            )
        ):
            raise ValueError(f"{path_name}: document why diffuseColor or normal can be uniform")
    return Project(
        str(cfg.get("name", root.name)),
        root,
        path,
        positive_int(cfg.get("reference_width", 1280), "reference_width"),
        {name: inside(root, file) for name, file in cases.items()},
        parsed,
        supports,
        uniform,
    )


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
