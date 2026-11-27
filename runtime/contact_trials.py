"""Headless, metered Isaac Lab contact experiments on the authored inspection assets.

Runs each scenario in a separate process. Neither moves a dynamic
payload by transform after reset. Reports simulator effective response, not real
material coefficients or hardware validation.
"""
import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial', required=True, choices=('rim', 'slide', 'grasp', 'open-control'))
    p.add_argument('--asset', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--dt', type=float, default=1/120)
    p.add_argument('--steps', type=int, default=360)
    p.add_argument('--speed', type=float, default=1.0, help='Rim probe initial lateral speed in m/s')
    p.add_argument('--finger-friction', type=float, default=1.5)
    p.add_argument('--rim-collider', default='RimRight', help='Relative collider prim within asset')
    p.add_argument('--slide-collider', default='Body', help='Relative contact collider prim within asset')
    p.add_argument('--grasp-left-collider', default='WallLeft')
    p.add_argument('--grasp-right-collider', default='WallRight')
    p.add_argument('--rim-probe-x', type=float, default=-0.12)
    p.add_argument('--rim-probe-z', type=float, default=0.034)
    p.add_argument('--rim-probe-radius', type=float, default=0.018)
    p.add_argument('--rim-precontact-x', type=float, default=0.20)
    p.add_argument('--rim-contact-zone-x', type=float, default=0.18)
    p.add_argument('--rim-max-x', type=float, default=0.25)
    p.add_argument('--finger-open-x', type=float, default=0.185)
    p.add_argument('--finger-closed-x', type=float, default=0.166)
    p.add_argument('--finger-center-z', type=float, default=0.09)
    p.add_argument('--finger-size-x', type=float, default=0.012)
    p.add_argument('--finger-size-y', type=float, default=0.10)
    p.add_argument('--finger-size-z', type=float, default=0.12)
    p.add_argument('--grasp-lift-m', type=float, default=0.18)
    p.add_argument('--support-top-z', type=float, default=0.0)
    from isaaclab.app import add_launcher_args
    add_launcher_args(p)
    supplied = list(sys.argv[1:] if argv is None else argv)
    a = p.parse_args(argv)
    a.device_explicit = any(x == '--device' or x.startswith('--device=') for x in supplied)
    if a.trial in ('grasp', 'open-control') and not a.device_explicit:
        a.device = 'cpu'
    a.asset = a.asset.resolve(); a.report = a.report.resolve()
    if not a.asset.is_file() or a.dt <= 0 or a.steps < 240 or a.speed <= 0 or a.finger_friction < 0:
        p.error('existing asset, positive dt/speed, nonnegative friction and at least 240 steps required')
    if not (a.rim_probe_x < a.rim_contact_zone_x < a.rim_precontact_x < a.rim_max_x):
        p.error('rim x values must increase from probe through zone/precontact to maximum')
    if a.rim_probe_radius <= 0 or a.finger_size_x <= 0 or a.finger_size_y <= 0 or a.finger_size_z <= 0:
        p.error('probe radius and finger dimensions must be positive')
    if not 0 < a.finger_closed_x < a.finger_open_x or a.grasp_lift_m <= 0:
        p.error('finger x centers and lift must be positive, with closed < open')
    for rel in (a.rim_collider, a.slide_collider, a.grasp_left_collider, a.grasp_right_collider):
        if not rel or rel.startswith('/') or '..' in rel.split('/'):
            p.error('collider paths must be relative descendants without ..')
    return a


def _version(name):
    try: return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: return 'unknown'


def _cube(stage, path, center, size, *, kinematic, friction=0.8, restitution=0.05, mass=None):
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade
    # Author physical dimensions in vertices so the fixture has no rigid-body
    # scale to interpret differently from its visual geometry.
    hx, hy, hz = (v / 2 for v in size)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(x, y, z) for x, y, z in (
        (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
        (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz))])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([
        0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4,
        1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7])
    mesh.CreateExtentAttr([Gf.Vec3f(-hx, -hy, -hz), Gf.Vec3f(hx, hy, hz)])
    mesh.CreateSubdivisionSchemeAttr('none')
    UsdGeom.Xformable(mesh.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*center))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
    if kinematic is not None:
        rb = UsdPhysics.RigidBodyAPI.Apply(mesh.GetPrim())
        rb.CreateKinematicEnabledAttr(kinematic)
    if mass is not None: UsdPhysics.MassAPI.Apply(mesh.GetPrim()).CreateMassAttr(mass)
    matpath = path + '_PhysicsMaterial'
    mat = UsdShade.Material.Define(stage, matpath)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(friction); pm.CreateDynamicFrictionAttr(friction)
    pm.CreateRestitutionAttr(restitution)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat, UsdShade.Tokens.weakerThanDescendants, 'physics')
    return mesh.GetPrim()


def _sphere(stage, path, center, radius, *, mass, friction, restitution):
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade
    sph = UsdGeom.Sphere.Define(stage, path)
    sph.CreateRadiusAttr(radius)
    UsdGeom.Xformable(sph.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*center))
    UsdPhysics.CollisionAPI.Apply(sph.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(sph.GetPrim())
    UsdPhysics.MassAPI.Apply(sph.GetPrim()).CreateMassAttr(mass)
    mat = UsdShade.Material.Define(stage, path + '_PhysicsMaterial')
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(friction); pm.CreateDynamicFrictionAttr(friction)
    pm.CreateRestitutionAttr(restitution)
    UsdShade.MaterialBindingAPI.Apply(sph.GetPrim()).Bind(mat, UsdShade.Tokens.weakerThanDescendants, 'physics')


def _view(pv, path):
    v = pv.create_rigid_body_view(path)
    if v is None or v.count != 1: raise RuntimeError(f'expected one PhysX rigid body at {path}')
    return v


def _pose(v):
    t = v.get_transforms().numpy()[0]
    w = v.get_velocities().numpy()[0]
    return {'p_m': [float(x) for x in t[:3]], 'q_xyzw': [float(x) for x in t[3:]],
            'v_mps': [float(x) for x in w[:3]], 'w_radps': [float(x) for x in w[3:]]}


def fit_deceleration(trace):
    """Fit the moving phase of a body's tensor vx trajectory to a line."""
    import numpy as np
    moving = [r for r in trace[5:] if r['v_mps'][0] > 0.10 and r['p_m'][0] < 0.35]
    if len(moving) < 8:
        return {'effective_dynamic_friction_fit': None, 'fit_slope_mps2': None,
                'fit_intercept_mps': None, 'fit_rmse_mps': None,
                'fit_samples': len(moving), 'fit_duration_s': None}
    times = np.array([r['t_s'] for r in moving])
    speeds = np.array([r['v_mps'][0] for r in moving])
    slope, intercept = np.polyfit(times, speeds, 1)
    residual = float(np.sqrt(np.mean((speeds-(slope*times+intercept))**2)))
    return {'effective_dynamic_friction_fit': float(-slope/9.81),
            'fit_slope_mps2': float(slope), 'fit_intercept_mps': float(intercept),
            'fit_rmse_mps': residual, 'fit_samples': len(moving),
            'fit_duration_s': float(times[-1]-times[0])}


def run(a, sim_cfg):
    import numpy as np
    import warp as wp
    import isaaclab.sim as sim_utils
    from isaaclab_physx.physics import PhysxManager
    from pxr import Usd, UsdPhysics
    started = time.monotonic()
    sim = sim_utils.SimulationContext(sim_cfg)
    stage = sim.stage
    source = Usd.Stage.Open(str(a.asset))
    if source is None or not source.GetDefaultPrim().IsValid(): raise RuntimeError('asset has no default prim')
    src = source.GetDefaultPrim()
    stage.DefinePrim('/World', 'Xform')
    cfg = sim_utils.UsdFileCfg(usd_path=str(a.asset))
    payload = cfg.func('/World/Payload', cfg)
    if not payload.IsValid() or not payload.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError('authored rigid payload failed to compose')
    authored_collider_count = sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(payload))
    required = ([a.rim_collider] if a.trial == 'rim' else
                [a.slide_collider] if a.trial == 'slide' else
                [a.grasp_left_collider, a.grasp_right_collider])
    for rel in required:
        collider = stage.GetPrimAtPath(str(payload.GetPath()) + '/' + rel)
        if not collider.IsValid() or not collider.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f'required authored collider missing: {rel}')
    # Spawn the authored component in an otherwise isolated stage. The tray is
    # kinematic for rim impact; the grasp support is a genuinely static collider.
    if a.trial == 'rim':
        UsdPhysics.RigidBodyAPI(payload).CreateKinematicEnabledAttr(True)
        _sphere(stage, '/World/Probe', (a.rim_probe_x, 0, a.rim_probe_z), a.rim_probe_radius,
                mass=0.03, friction=0.12, restitution=0.35)
    elif a.trial == 'slide':
        _cube(stage, '/World/Support', (0, 0, a.support_top_z-0.025), (1.0, 0.8, 0.05),
              kinematic=True, friction=0.8, restitution=0.0)
    else:
        _cube(stage, '/World/Support', (0, 0, a.support_top_z-0.025), (0.8, 0.8, 0.05),
              kinematic=None, friction=0.8)
        # 120 mm tall pads span the carton side walls; independent kinematic
        # bodies have no attachment/joint/parenting relationship to payload.
        for side, x in [('L', -a.finger_open_x), ('R', a.finger_open_x)]:
            _cube(stage, '/World/Finger'+side, (x, 0, a.finger_center_z),
                  (a.finger_size_x, a.finger_size_y, a.finger_size_z),
                  kinematic=True, friction=a.finger_friction)
    sim.reset()
    pv = PhysxManager.get_physics_sim_view()
    if pv is None: raise RuntimeError('PhysX tensor view missing')
    _view(pv, '/World/Payload')
    target_path = '/World/Probe' if a.trial == 'rim' else '/World/Payload'
    target = _view(pv, target_path)
    if a.trial in ('rim', 'slide'):
        existing_velocity = target.get_velocities()
        velocity = existing_velocity.numpy().copy()
        velocity[0, 0] = a.speed
        tensor_device = existing_velocity.device
        target.set_velocities(
            wp.array(velocity, dtype=wp.float32, device=tensor_device),
            wp.array([0], dtype=wp.int32, device=tensor_device),
        )
        if abs(_pose(target)['v_mps'][0] - a.speed) > 0.05:
            raise RuntimeError('PhysX did not apply initial lateral velocity')
    else:
        fingers = [_view(pv, '/World/FingerL'), _view(pv, '/World/FingerR')]
        finger_devices = [v.get_transforms().device for v in fingers]
        finger_indices = [wp.array([0], dtype=wp.int32, device=d) for d in finger_devices]
        post_reset_payload_state = _pose(target)
        post_reset_finger_states = [_pose(v) for v in fingers]
        # Reset may leave kinematic tensor poses at the origin with velocities
        # toward their authored USD positions. Teleport to the open pose and
        # set an identical target before the first counted physics step.
        for side, v, device, index in zip((-1, 1), fingers, finger_devices, finger_indices):
            pose = wp.array([[side*a.finger_open_x, 0, a.finger_center_z, 0, 0, 0, 1]],
                            dtype=wp.float32, device=device)
            v.set_transforms(pose, index)
            v.set_kinematic_targets(pose, index)
        # PhysX calculates kinematic velocities from successive targets. The
        # CPU API rejects set_velocities on these rigid bodies.
        for _ in range(2):
            sim.step()
        payload_device = target.get_transforms().device
        payload_index = wp.array([0], dtype=wp.int32, device=payload_device)
        payload_pose = wp.array([[0, 0, a.support_top_z, 0, 0, 0, 1]], dtype=wp.float32, device=payload_device)
        target.set_transforms(payload_pose, payload_index)
        target.set_velocities(wp.zeros((1, 6), dtype=wp.float32, device=payload_device), payload_index)
        initial_payload_state = _pose(target)
        initial_finger_states = [_pose(v) for v in fingers]
        pose_error = max(abs(q - r) for q, r in zip(initial_payload_state['p_m'], (0, 0, a.support_top_z)))
        speed_error = max(abs(x) for x in initial_payload_state['v_mps'] + initial_payload_state['w_radps'])
        for side, state in zip((-1, 1), initial_finger_states):
            pose_error = max(pose_error, abs(state['p_m'][0]-side*a.finger_open_x),
                             abs(state['p_m'][1]), abs(state['p_m'][2]-a.finger_center_z))
            speed_error = max(speed_error, *(abs(x) for x in state['v_mps'] + state['w_radps']))
        if pose_error > 1e-4 or speed_error > 1e-3:
            a.report.parent.mkdir(parents=True, exist_ok=True)
            a.report.write_text(json.dumps({
                'status': 'startup_failed', 'trial': a.trial,
                'stationary_finger_steps': 2,
                'post_reset_payload_state': post_reset_payload_state,
                'post_reset_finger_states': post_reset_finger_states,
                'initial_payload_state': initial_payload_state,
                'initial_finger_states': initial_finger_states,
                'pose_error_m': pose_error, 'velocity_error': speed_error,
            }, indent=2)+'\n')
            raise RuntimeError(f'{a.trial} startup pose/velocity mismatch; report: {a.report}')
    start_steps = sim.get_physics_step_count()
    trace = []
    for i in range(a.steps):
        if not sim.is_running() or not sim.is_playing():
            raise RuntimeError(f'simulation stopped at step {i}')
        if a.trial not in ('rim', 'slide'):
            # close 0-.30 s, lift .30-1.50 s, hold 1.50-2.0 s,
            # release at 2.0 s. The open control stays beyond contact.
            t = i*a.dt
            close = min(1., t/.30)
            zlift = a.grasp_lift_m * min(1., max(0., (t-.30)/1.20))
            opening = min(1., max(0., (t-2.0)/.20))
            for side, v, device, index in zip((-1, 1), fingers, finger_devices, finger_indices):
                xabs = (a.finger_open_x if a.trial == 'open-control' else
                        a.finger_open_x - (a.finger_open_x-a.finger_closed_x)*close +
                        (a.finger_open_x-a.finger_closed_x)*opening)
                target_pose = wp.array([[side*xabs, 0, a.finger_center_z+zlift, 0, 0, 0, 1]],
                                       dtype=wp.float32, device=device)
                v.set_kinematic_targets(target_pose, index)
        sim.step()
        target_state = _pose(target)
        row = {'step': i+1, 't_s': (i+1)*a.dt, **target_state}
        if a.trial not in ('rim', 'slide'):
            row['fingers_p_m'] = [_pose(v)['p_m'] for v in fingers]
        trace.append(row)
    if sim.get_physics_step_count()-start_steps != a.steps:
        raise RuntimeError('physics step count mismatch')
    if a.trial == 'rim':
        pre = [r for r in trace if r['p_m'][0] < a.rim_precontact_x and r['v_mps'][0] > 0.05]
        hit = [r for r in trace if r['p_m'][0] > a.rim_contact_zone_x]
        post = [r for r in trace if r['p_m'][0] > a.rim_contact_zone_x and r['v_mps'][0] < -0.05]
        v_in = pre[-1]['v_mps'][0] if pre else 0.
        v_out = post[0]['v_mps'][0] if post else 0.
        metrics = {'max_x_m': max(r['p_m'][0] for r in trace),
                   'inbound_vx_mps': v_in, 'outbound_vx_mps': v_out,
                   'effective_rebound_ratio': abs(v_out)/v_in if v_in else None,
                   'wall_zone_samples': len(hit),
                   'contact_material_pair': [f'asset:{a.rim_collider}', 'probe sphere (friction 0.12, restitution 0.35)'],
                   'nonrebound': not bool(post),
                   'inbound_step': pre[-1]['step'] if pre else None,
                   'outbound_step': post[0]['step'] if post else None,
                   'probe_final_z_m': trace[-1]['p_m'][2]}
        passed = bool(hit and post and metrics['max_x_m'] < a.rim_max_x)
    elif a.trial == 'slide':
        # Initial deceleration while translating on the tray base. Effective
        # coefficient includes solver and contact effects, not a measured real value.
        fit = fit_deceleration(trace)
        metrics = {'initial_vx_mps': trace[0]['v_mps'][0],
                   'final_vx_mps': trace[-1]['v_mps'][0],
                   'travel_x_m': trace[-1]['p_m'][0]-trace[0]['p_m'][0],
                   **fit,
                   'fit_condition': 'steps after first 5; vx > 0.10 m/s; x < 0.35 m',
                   'acceptance': {'mu_min': 0.02, 'mu_max': 1.5,
                                  'rmse_max_mps': 0.15, 'min_samples': 8},
                   'contact_material_pair': [f'asset:{a.slide_collider}', 'support (friction 0.8, restitution 0.0)'],
                   'max_x_m': max(r['p_m'][0] for r in trace)}
        mu = fit['effective_dynamic_friction_fit']
        residual = fit['fit_rmse_mps']
        passed = bool(mu is not None and 0.02 < mu < 1.5 and residual < 0.15 and
                      metrics['travel_x_m'] > 0.005 and metrics['max_x_m'] < 0.45)
    else:
        # Carton bottom starts at z=0. A successful lift must be sustained
        # after finger ascent, not just a brief collision impulse.
        hold = [r for r in trace if 1.65 <= r['t_s'] <= 1.95]
        lift = [r for r in trace if 1.45 <= r['t_s'] <= 1.95]
        mean_hold = sum(r['p_m'][2] for r in hold)/len(hold)
        release = trace[-1]['p_m'][2]
        metrics = {'max_payload_z_m': max(r['p_m'][2] for r in trace),
                   'mean_hold_payload_z_m': mean_hold,
                   'min_lift_payload_z_m': min(r['p_m'][2] for r in lift),
                   'final_payload_z_m': release,
                   'hold_tail_vertical_span_m': max(r['p_m'][2] for r in hold)-min(r['p_m'][2] for r in hold),
                   'release_fall_m': mean_hold-release}
        if a.trial == 'grasp':
            passed = (metrics['min_lift_payload_z_m'] > 0.08 and
                      metrics['hold_tail_vertical_span_m'] < 0.04 and
                      metrics['release_fall_m'] > 0.06)
        else:
            passed = metrics['max_payload_z_m'] < 0.04
    result = {'status': 'passed' if passed else 'failed', 'trial': a.trial,
              'stationary_finger_steps': 2 if a.trial in ('grasp', 'open-control') else 0,
              'asset': str(a.asset), 'asset_default_prim': str(src.GetPath()),
              'required_colliders': required, 'authored_collider_count': authored_collider_count,
              'geometry_inputs': {key: getattr(a, key) for key in (
                  'rim_probe_x', 'rim_probe_z', 'rim_probe_radius', 'rim_precontact_x',
                  'rim_contact_zone_x', 'rim_max_x', 'finger_open_x', 'finger_closed_x',
                  'finger_center_z', 'finger_size_x', 'finger_size_y', 'finger_size_z',
                  'grasp_lift_m', 'support_top_z')},
              'physics_backend': type(sim.cfg.physics).__name__, 'device': a.device,
              'device_explicit': a.device_explicit,
              'dt_s': a.dt, 'steps': a.steps, 'physics_steps_start': start_steps,
              'physics_steps_end': sim.get_physics_step_count(),
              'speed_mps': a.speed if a.trial in ('rim', 'slide') else None,
              'finger_friction_authored': a.finger_friction if a.trial not in ('rim', 'slide') else None,
              'metric_scope': 'effective simulated contact response; no real-world coefficient fit',
              'tensor_source': 'omni.physics.tensors rigid_body_view',
              'post_reset_payload_state': post_reset_payload_state if a.trial in ('grasp', 'open-control') else None,
              'post_reset_finger_states': post_reset_finger_states if a.trial in ('grasp', 'open-control') else None,
              'initial_payload_state': initial_payload_state if a.trial in ('grasp', 'open-control') else None,
              'initial_finger_states': initial_finger_states if a.trial in ('grasp', 'open-control') else None,
              'metrics': metrics, 'trace': trace,
              'wall_elapsed_s': time.monotonic()-started,
              'versions': {'isaaclab': _version('isaaclab')}}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'trace'}, indent=2))
    if not passed: raise RuntimeError(f'{a.trial} acceptance threshold failed; report: {a.report}')


def main(argv=None):
    a = parse_args(argv)
    import isaaclab.sim as sim_utils
    from isaaclab.app import launch_simulation
    from isaaclab_physx.physics import PhysxCfg
    cfg = sim_utils.SimulationCfg(dt=a.dt, device=a.device, physics=PhysxCfg())
    with launch_simulation(cfg, a):
        run(a, cfg)


if __name__ == '__main__': sys.exit(main())
