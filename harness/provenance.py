"""Verify the recorded input and capture-code SHA-256 against files or a requested Git revision."""

import argparse
import hashlib
import subprocess
from pathlib import Path

from .campaign import digest, load, read, write


def git_output(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.PIPE
    ).strip()


def verify_source(capture, config, revision=None, code_revision=None):
    header = read(capture / "campaign.json")
    base = Path(__file__).resolve().parents[1]
    config_root, cfg = load(config)
    root = (config_root / cfg["stage"]).resolve().parent
    manifest = {
        ((root / s["path"]).resolve(), s["path"]): s["sha256"]
        for s in header["sources"]
    }
    modules = set()
    for name, expected in header["code"].items():
        paths = [
            p for p in (base / "runtime" / name, base / "harness" / name) if p.is_file()
        ]
        if len(paths) != 1:
            raise ValueError(f"cannot identify capture module {name}")
        manifest[(paths[0], name)] = expected
        modules.add(paths[0])

    def resolve(ref):
        return (
            git_output(
                base, "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"
            )
            if ref
            else None
        )

    commit = resolve(revision)
    code = resolve(code_revision) if code_revision else commit
    gitroot = (
        Path(git_output(base, "rev-parse", "--show-toplevel"))
        if commit or code
        else None
    )
    rows = []
    for (path, label), expected in manifest.items():
        measured = None
        mode = "file"
        source_revision = code if path in modules else commit
        if source_revision and path.is_relative_to(gitroot):
            mode = "git"
            with subprocess.Popen(
                [
                    "git",
                    "-C",
                    str(gitroot),
                    "cat-file",
                    "blob",
                    source_revision + ":" + str(path.relative_to(gitroot)),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            ) as proc:
                measured = hashlib.file_digest(proc.stdout, "sha256").hexdigest()
                if proc.wait():
                    measured = None
        elif path.is_file():
            measured = digest(path)
        rows.append(
            {
                "input": label,
                "read_from": mode,
                "sha256": expected,
                "observed": measured,
                "matches": measured == expected,
            }
        )
    return {
        "campaign_key": header["campaign_key"],
        "commit": commit,
        "capture_code_commit": code,
        "items": rows,
        "status": "pass" if all(row["matches"] for row in rows) else "fail",
    }


def main(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture", type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--git-rev", help="Asset and configuration revision")
    p.add_argument(
        "--code-rev",
        help="Separate capture-code revision, if the scene was calibrated later",
    )
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    data = verify_source(a.capture, a.config, a.git_rev, a.code_rev)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write(a.out, data)
    print(f"{data['status']} ({len(data['items'])} inputs)")
    return int(data["status"] != "pass")
