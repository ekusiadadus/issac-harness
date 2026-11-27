"""Independently check whether a sampled mesh surface is visible in the native radial range."""

import numpy as np


class DescendantParking:
    """Keep target geometry registered while excluding an edited child from views."""

    def __init__(self, stage, camera):
        self.stage = stage
        self.camera = camera
        self._active = {}
        self._created_ops = {}

    def _disarm(self, path):
        from pxr import UsdGeom

        for root, session_order in reversed(self._active.pop(path, [])):
            prim = self.stage.GetPrimAtPath(root)
            if not prim:
                continue
            attr = UsdGeom.Xformable(prim).GetXformOpOrderAttr()
            if session_order is None:
                attr.Clear()
            else:
                attr.Set(session_order)

    def restore(self):
        """Disarm before variant changes or geometric bounds/probe calculations."""
        for path in list(self._active):
            self._disarm(path)

    def _roots(self, prim):
        from pxr import Usd, UsdGeom

        roots = []
        for child in Usd.PrimRange(prim):
            if not child.IsA(UsdGeom.Xformable):
                continue
            xf = UsdGeom.Xformable(child)
            if child == prim or xf.GetResetXformStack() or not any(
                child.GetPath().HasPrefix(root.GetPath()) for root in roots
            ):
                roots.append(child)
        if not roots:
            raise ValueError(
                f"target visibility edit has no Xformable geometry: {prim.GetPath()}"
            )
        return roots

    def _park(self, prim):
        from pxr import Gf, Sdf, Usd, UsdGeom

        path = str(prim.GetPath())
        xf = UsdGeom.Xformable(prim)
        order_path = Sdf.Path(path).AppendProperty("xformOpOrder")
        session_spec = self.stage.GetSessionLayer().GetAttributeAtPath(order_path)
        session_order = (
            list(session_spec.default)
            if session_spec and session_spec.default is not None
            else None
        )
        now = Usd.TimeCode.Default()
        current = xf.ComputeLocalToWorldTransform(now)
        desired = Gf.Matrix4d(current)
        camera_world = UsdGeom.Xformable(
            self.camera.GetPrim()
        ).ComputeLocalToWorldTransform(now)
        cache = UsdGeom.BBoxCache(now, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        corners = (
            [bounds.GetCorner(i) for i in range(8)]
            if not bounds.IsEmpty()
            else [current.ExtractTranslation()]
        )
        radius = max((corner - current.ExtractTranslation()).GetLength() for corner in corners)
        distance = max(50.0, 4.0 * radius)
        desired.SetTranslateOnly(camera_world.Transform(Gf.Vec3d(0, 0, distance)))
        parent = (
            Gf.Matrix4d(1)
            if xf.GetResetXformStack()
            else xf.ComputeParentToWorldTransform(now)
        )
        local = xf.GetLocalTransformation()
        # USD's final xformOp is applied on the left of the previous local ops.
        offset = desired * parent.GetInverse() * local.GetInverse()
        suffix = self._created_ops.get(path)
        if suffix is None:
            suffix = "harnessVisibilityPark"
            count = 0
            while prim.GetAttribute("xformOp:transform:" + suffix):
                count += 1
                suffix = f"harnessVisibilityPark{count}"
            self._created_ops[path] = suffix
        existing = prim.GetAttribute("xformOp:transform:" + suffix)
        if existing:
            op = UsdGeom.XformOp(existing)
            xf.SetXformOpOrder([*xf.GetOrderedXformOps(), op], xf.GetResetXformStack())
        else:
            op = xf.AddTransformOp(opSuffix=suffix)
        op.Set(offset)
        actual = xf.ComputeLocalToWorldTransform(now)
        if not np.allclose(np.asarray(actual), np.asarray(desired), atol=1e-7):
            raise ValueError(f"visibility parking transform readback failed: {path}")
        return path, session_order

    def set(self, path, visible):
        from pxr import UsdGeom

        prim = self.stage.GetPrimAtPath(path)
        if not prim:
            raise ValueError(f"missing target visibility prim: {path}")
        attr = UsdGeom.Imageable(prim).GetVisibilityAttr()
        self._disarm(path)
        # Both states keep the prim visible to Hydra. An inactive child is moved
        # behind the camera rather than removed from scene registration.
        attr.Set(UsdGeom.Tokens.inherited)
        if visible:
            return str(attr.Get())
        roots = self._roots(prim)
        self._active[path] = []
        for root in roots:
            self._active[path].append(self._park(root))
        return "parked"


def depth_witness(points_local, world, camera, depth):
    if not len(points_local):
        return 0
    points = np.c_[points_local, np.ones(len(points_local))] @ np.asarray(world)
    # USD cameras look along -Z, optical image rows increase downwards.
    xyz = (points @ np.linalg.inv(np.asarray(camera["matrix_row"])))[:, :3] * [
        1,
        -1,
        -1,
    ]
    xyz = xyz[xyz[:, 2] > 0.01]
    if not len(xyz):
        return 0
    K = np.array(camera["K"]).reshape(3, 3)
    uv = xyz @ K.T
    px, py = np.rint(uv[:, :2] / uv[:, 2:3]).astype(int).T
    yy, xx = depth.shape
    in_frame = (px >= 2) & (px < xx - 2) & (py >= 2) & (py < yy - 2)
    px, py, xyz = px[in_frame], py[in_frame], xyz[in_frame]
    measured = depth[py, px]
    expected = np.linalg.norm(xyz, axis=1)
    match = np.isfinite(measured) & (
        np.abs(measured - expected) <= np.maximum(0.002, expected * 0.0015)
    )
    return len(np.unique(py[match] * xx + px[match]))
