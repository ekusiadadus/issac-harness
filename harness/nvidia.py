"""Use NVIDIA's unmodified default USD validation rules in their supported USD environment."""

import json
import subprocess

from .project import file_hash, write_json

USD_TOOL = "usd-validation-nvidia[usd]==1.23.0"
PROFILE = "Robotics-Prop"
PROFILE_VERSION = "5.0.0"


def validate(project, report, directory):
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    # The CLI's result lists findings, so a clean run has an empty "rules" array.
    # Resolve its default set from the same pinned package and pass those names explicitly.
    script = "import json; from usd_validation_nvidia import ValidationEngine; print(json.dumps(sorted(r.__name__ for r in ValidationEngine().rules)))"
    rules = json.loads(
        subprocess.check_output(
            ["uv", "run", "--no-project", "--python", "3.12", "--with", USD_TOOL, "python", "-c", script],
            text=True,
        )
    )
    if not rules:
        raise ValueError("NVIDIA USD rule registry returned no enabled rules")
    write_json(directory / "enabled-rules.json", rules)
    for name, case in report["cases"].items():
        output = directory / (name + ".json")
        log = directory / (name + ".log")
        with log.open("w") as stream:
            result = subprocess.run(
                [
                    "uvx",
                    "--python",
                    "3.12",
                    "--from",
                    USD_TOOL,
                    "nvidia_usd_validate",
                    str(project.cases[name]),
                    "--no-init-rules",
                    *(argument for rule in rules for argument in ("--rule", rule)),
                    "--no-fix",
                    "--no-stamp",
                    "--json-output",
                    str(output),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        data = json.loads(output.read_text()) if output.is_file() else {}
        passed = result.returncode == 0 and data.get("status") == "PASS"
        issues = {r["rule"]["name"]: len(r["issues"]) for r in data.get("rules", []) if r.get("issues")}
        case["nvidia_validation"] = {
            "version": "1.23.0",
            "enabled_rules": rules,
            "status": data.get("status", "error"),
            "exit_code": result.returncode,
            "issue_counts": issues,
        }
        if not passed:
            case["errors"].append(f"NVIDIA validation failed {issues}. See {log}")
            case["status"] = report["status"] = "fail"


def simready(assets, directory):
    directory = directory.resolve()
    if len({asset.stem for asset in assets}) != len(assets):
        raise ValueError("each asset needs a distinct output name")
    directory.mkdir(parents=True, exist_ok=False)
    results = []
    for path in assets:
        path = path.resolve(strict=True)
        output = directory / (path.stem + ".json")
        log = directory / (path.stem + ".log")
        cmd = [
            "uv",
            "run",
            "--no-project",
            "--python",
            "3.12",
            "--with",
            "simready-validate==2026.10.0",
            "--with",
            "simready-foundation-tier-core==2026.10.0",
            "--with",
            "usd-core==26.11",
            "simready-validate",
            str(path),
            "--profile",
            PROFILE,
            "--version",
            PROFILE_VERSION,
            "--output",
            str(output),
        ]
        with log.open("w") as stream:
            proc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
        entries = json.loads(output.read_text()) if output.is_file() else {}
        entry = entries.get(str(path), {})
        features = entry.get("features_summary", {})
        blocking = [name for name, item in features.items() if item.get("blocking_failure")]
        passed = (
            bool(features)
            and entry.get("profile_id") == PROFILE
            and entry.get("profile_version") == PROFILE_VERSION
            and proc.returncode == 0
            and not blocking
        )
        print(
            f"{'PASS' if passed else 'FAIL'} {path.name} · {len(features)} features · {len(blocking)} blocking · {log}"
        )
        results.append(
            dict(
                asset=path.name,
                sha256=file_hash(path),
                features=len(features),
                blocking=blocking,
                status="pass" if passed else "fail",
                exit_code=proc.returncode,
                report=output.name,
            )
        )
    passed = all(row["status"] == "pass" for row in results)
    write_json(
        directory / "summary.json",
        dict(profile=PROFILE, version=PROFILE_VERSION, assets=results, status="pass" if passed else "fail"),
    )
    return passed
