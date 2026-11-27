"""Execute an explicit campaign plan on one or more workers sharing an output directory."""

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .campaign import digest, inspect_frame, read, write


def validate_plan(plan, config):
    if (
        plan["status"] != "planned"
        or plan["config_sha256"] != digest(config)
        or plan["seed"] < 0
    ):
        raise ValueError("plan does not belong to this configuration")
    offset = 0
    for shard in plan["shards"]:
        if (
            shard["start"] != offset
            or type(shard["count"]) is not int
            or shard["count"] <= 0
        ):
            raise ValueError("shards must be contiguous, disjoint and nonempty")
        offset += shard["count"]
    if offset != plan["frames"] or not offset:
        raise ValueError("plan length mismatch")


def execute(
    config,
    plan_path,
    root,
    worker=0,
    workers=1,
    retries=1,
    through=None,
    render_args=(),
    runner=None,
):
    plan = read(plan_path)
    validate_plan(plan, config)
    if (
        workers < 1
        or not 0 <= worker < workers
        or retries < 0
        or (through is not None and through < 1)
    ):
        raise ValueError("invalid worker count, index, attempt limit or shard count")
    protected = ("--seed", "--start", "--count", "--out", "--config")
    if any(x.split("=")[0] in protected for x in render_args):
        raise ValueError("the plan owns seed, index, count and output")
    if runner is None:
        runner = subprocess.call
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "jobs" / f"worker-{worker:04d}-{uuid4().hex[:12]}"
    logs.mkdir(parents=True)
    assigned = [(i, s) for i, s in enumerate(plan["shards"]) if i % workers == worker]
    if through is not None:
        assigned = assigned[:through]
    receipt = {
        "worker": worker,
        "workers": workers,
        "plan_sha256": digest(plan_path),
        "shards": [],
        "status": "running",
    }
    write(logs / "run.json", receipt)
    for index, shard in assigned:
        cmd = [
            sys.executable,
            "-m",
            "harness",
            "campaign",
            "capture",
            str(config.resolve()),
            "--out",
            str(root),
            "--start",
            str(shard["start"]),
            "--count",
            str(shard["count"]),
            "--seed",
            str(plan["seed"]),
            *render_args,
        ]
        item = {"shard_index": index, **shard, "attempts": []}
        receipt["shards"].append(item)
        completed = False
        for attempt in range(retries + 1):
            name = f"shard-{index:06d}-try-{attempt + 1}.log"
            with (logs / name).open("w") as stream:
                code = runner(cmd, stdout=stream, stderr=subprocess.STDOUT)
            item["attempts"].append({"log": name, "exit_code": code})
            if code == 0:
                head = read(root / "campaign.json")
                if head["seed"] != plan["seed"]:
                    raise ValueError("output seed differs from plan")
                for n in range(shard["start"], shard["start"] + shard["count"]):
                    inspect_frame(root / "frames" / f"{n:08d}", head, pixels=False)
                completed = True
                break
        item["verified_frames"] = shard["count"] if completed else 0
        if not completed:
            receipt["status"] = "failed"
            write(logs / "run.json", receipt)
            raise RuntimeError(f"capture failed, see {logs / name}")
        write(logs / "run.json", receipt)
    receipt["status"] = "worker_complete"
    receipt["finished_utc"] = datetime.now(UTC).isoformat()
    write(logs / "run.json", receipt)
    return logs / "run.json"


def main(argv):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("config", type=Path)
    cli.add_argument("--plan", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--worker-index", type=int, default=0)
    cli.add_argument("--worker-count", type=int, default=1)
    cli.add_argument("--max-shards", type=int)
    cli.add_argument("--retries", type=int, default=1)
    cli.add_argument("--asset-root", type=Path)
    args, rest = cli.parse_known_args(argv)
    if args.asset_root:
        rest += ["--asset-root", str(args.asset_root)]
    print(
        execute(
            args.config,
            args.plan,
            args.out,
            args.worker_index,
            args.worker_count,
            args.retries,
            args.max_shards,
            rest,
        )
    )
