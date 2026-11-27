"""Bounded Isaac Lab v3.0 USD spawn and physics-step smoke for a local asset.

Run with Isaac Lab's Python environment. --marker-prim is a source USD prim
under its default prim, so the check proves an authored descendant was loaded.
"""

import argparse
import importlib.metadata
import json
import math
import sys
from pathlib import Path


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True, type=Path, help="Local asset-only USD file")
    parser.add_argument("--spawn-prim", default="/World/SmokeAsset", help="Destination root prim")
    parser.add_argument("--marker-prim", required=True, help="Authored descendant path in source asset")
    parser.add_argument("--steps", type=positive_int, default=10)
    parser.add_argument("--drop-prim", help="Source rigid-body path to lift and measure settling")
    parser.add_argument("--drop-height", type=float, default=0.06, help="Lift in meters")
    parser.add_argument("--settle-tolerance", type=float, default=0.025, help="Final height error in meters")
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--report", type=Path, help="Optional JSON result path")
    from isaaclab.app import add_launcher_args

    add_launcher_args(parser)
    args = parser.parse_args(argv)
    args.asset = args.asset.resolve()
    if args.report:
        args.report = args.report.resolve()
    if not args.asset.is_file() or args.asset.suffix.lower() not in (".usd", ".usda", ".usdc"):
        parser.error("--asset must be an existing local USD file")
    if not args.spawn_prim.startswith("/") or args.spawn_prim == "/" or not args.marker_prim.startswith("/"):
        parser.error("prim paths must be absolute")
    if args.dt <= 0:
        parser.error("--dt must be positive")
    if args.drop_prim and (not args.drop_prim.startswith("/") or args.drop_height <= 0 or args.steps < 60):
        parser.error("drop check requires absolute --drop-prim, positive height, and at least 60 steps")
    return args


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def run(args, sim_cfg):
    import isaaclab.sim as sim_utils
    from isaacsim.core.version import get_version
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    # Create the simulation stage before opening the asset for source inspection;
    # SimulationContext can otherwise select an unrelated cached USD stage.
    sim = sim_utils.SimulationContext(sim_cfg)
    stage = sim.stage
    source = Usd.Stage.Open(str(args.asset))
    if source is None:
        raise RuntimeError(f"USD could not open asset: {args.asset}")
    default = source.GetDefaultPrim()
    if not default.IsValid():
        raise RuntimeError("asset USD requires a defaultPrim for deterministic UsdFileCfg spawning")
    marker_path = Sdf.Path(args.marker_prim)
    if (
        not marker_path.IsAbsolutePath()
        or not marker_path.HasPrefix(default.GetPath())
        or marker_path == default.GetPath()
    ):
        raise RuntimeError(f"--marker-prim must be a descendant of defaultPrim {default.GetPath()}")
    source_marker = source.GetPrimAtPath(marker_path)
    if not source_marker.IsValid() or not source_marker.IsActive():
        raise RuntimeError(f"marker is not an active authored prim in asset: {marker_path}")
    marker_type = source_marker.GetTypeName()
    if not marker_type:
        raise RuntimeError("marker must have an authored prim type")

    stage.DefinePrim("/World", "Xform")
    if stage.GetPrimAtPath(args.spawn_prim).IsValid():
        raise RuntimeError(f"spawn destination already exists: {args.spawn_prim}")
    cfg = sim_utils.UsdFileCfg(usd_path=str(args.asset))
    spawned = cfg.func(args.spawn_prim, cfg)
    if not spawned.IsValid():
        raise RuntimeError("UsdFileCfg returned an invalid prim")

    # A wrapper alone can exist even if its reference failed to compose.
    relative_marker = marker_path.MakeRelativePath(default.GetPath())
    loaded_path = Sdf.Path(args.spawn_prim).AppendPath(relative_marker)
    loaded = stage.GetPrimAtPath(loaded_path)
    if not loaded.IsValid() or not loaded.IsActive() or loaded.GetTypeName() != marker_type:
        raise RuntimeError(f"authored marker did not compose at {loaded_path} (expected {marker_type})")

    drop_data = None
    if args.drop_prim:
        source_drop = Sdf.Path(args.drop_prim)
        if not source_drop.HasPrefix(default.GetPath()):
            raise RuntimeError("--drop-prim must be under asset defaultPrim")
        loaded_drop = stage.GetPrimAtPath(
            Sdf.Path(args.spawn_prim).AppendPath(source_drop.MakeRelativePath(default.GetPath()))
        )
        if not loaded_drop or not loaded_drop.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"--drop-prim is not a loaded rigid body: {loaded_drop.GetPath()}")
        transform = UsdGeom.Xformable(loaded_drop)
        before = transform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()[2]
        lift = transform.AddTranslateOp(opSuffix="smokeDrop")
        lift.Set(Gf.Vec3d(0, 0, args.drop_height))
        lifted = transform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()[2]
        if lifted - before < args.drop_height * 0.9:
            raise RuntimeError("lifting rigid body did not change its world height")
        drop_data = {"prim": str(loaded_drop.GetPath()), "rest_z_m": before, "lifted_z_m": lifted}

    sim.reset()
    if drop_data:
        from isaaclab_physx.physics import PhysxManager

        physics_view = PhysxManager.get_physics_sim_view()
        if physics_view is None:
            raise RuntimeError("PhysX tensor view was not initialized after reset")
        body_view = physics_view.create_rigid_body_view(str(loaded_drop.GetPath()))
        if body_view is None or body_view.count != 1:
            raise RuntimeError(f"PhysX did not register exactly one rigid body at {loaded_drop.GetPath()}")

        def body_reading():
            pose = body_view.get_transforms().numpy()
            velocity = body_view.get_velocities().numpy()
            return float(pose[0, 2]), [float(v) for v in velocity[0]]

        initialized_z, initialized_velocity = body_reading()
        drop_data["tensor_initial_z_m"] = initialized_z
        drop_data["tensor_initial_velocity_mps"] = initialized_velocity[:3]
        if abs(initialized_z - drop_data["lifted_z_m"]) > 0.015:
            raise RuntimeError(f"PhysX body initial pose disagrees with authored lift: {drop_data}")
    start_steps = sim.get_physics_step_count()
    observed_z = []
    observed_v = []
    for index in range(args.steps):
        if not sim.is_running() or not sim.is_playing():
            raise RuntimeError(f"simulation stopped before step {index + 1}")
        sim.step()
        if drop_data:
            z, velocity = body_reading()
            observed_z.append(z)
            observed_v.append(velocity)
    end_steps = sim.get_physics_step_count()
    if end_steps - start_steps != args.steps:
        raise RuntimeError(
            f"physics step count mismatch: {start_steps} -> {end_steps}, expected {args.steps}"
        )
    if not stage.GetPrimAtPath(loaded_path).IsValid():
        raise RuntimeError("authored marker disappeared during stepping")
    if drop_data:
        final = observed_z[-1]
        tail = observed_z[-min(30, len(observed_z)) :]
        drop_data.update(
            {
                "final_z_m": final,
                "fall_m": drop_data["lifted_z_m"] - final,
                "rest_error_m": abs(final - drop_data["rest_z_m"]),
                "tail_range_m": max(tail) - min(tail),
                "final_linear_velocity_mps": observed_v[-1][:3],
                "final_angular_velocity_radps": observed_v[-1][3:],
                "tail_max_speed_mps": max(math.hypot(*v[:3]) for v in observed_v[-30:]),
                "tail_max_angular_speed_radps": max(math.hypot(*v[3:]) for v in observed_v[-30:]),
                "trace": [
                    {"step": n + 1, "z_m": observed_z[n], "vz_mps": observed_v[n][2]}
                    for n in range(args.steps)
                    if n < 20 or (n + 1) % 10 == 0 or n + 1 == args.steps
                ],
                "tensor_source": "omni.physics.tensors rigid_body_view",
            }
        )
        if drop_data["fall_m"] < args.drop_height * 0.5:
            raise RuntimeError(f"rigid body did not fall: {drop_data}")
        if (
            drop_data["rest_error_m"] > args.settle_tolerance
            or drop_data["tail_range_m"] > 0.01
            or drop_data["tail_max_speed_mps"] > 0.01
            or drop_data["tail_max_angular_speed_radps"] > 0.05
        ):
            raise RuntimeError(f"rigid body did not settle on support: {drop_data}")
    report = {
        "status": "passed",
        "asset": str(args.asset),
        "asset_default_prim": str(default.GetPath()),
        "source_marker": str(marker_path),
        "loaded_marker": str(loaded_path),
        "marker_type": marker_type,
        "spawn_prim": args.spawn_prim,
        "steps": args.steps,
        "dt": args.dt,
        "physics_steps_start": start_steps,
        "physics_steps_end": end_steps,
        "physics_backend": type(sim.cfg.physics).__name__,
        "drop_settle": drop_data,
        "versions": {"isaaclab": version("isaaclab"), "isaacsim_app": list(get_version())},
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main(argv=None):
    args = parse_args(argv)
    import isaaclab.sim as sim_utils
    from isaaclab.app import launch_simulation
    from isaaclab_physx.physics import PhysxCfg

    sim_cfg = sim_utils.SimulationCfg(dt=args.dt, device=args.device, physics=PhysxCfg())
    with launch_simulation(sim_cfg, args):
        run(args, sim_cfg)


if __name__ == "__main__":
    sys.exit(main())
