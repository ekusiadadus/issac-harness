# Texture scale gate

Run this check on each mesh whose diffuse bitmap is deliberately tiled. It reads the selected USD mesh, its bound preview material, the float2 UV primvar and image dimensions. It writes a JSON report and exits nonzero if the mapping cannot be assessed or a required condition fails.

```sh
uv run python -m harness.texel_check examples/campaign/families/scene.usda \
  --mesh /World/Surface --out /tmp/texture-scale.json \
  --min-texels-per-metre 390
```

Repeat `--mesh` for other target meshes. The optional minimum is the image pixels per metre in the least detailed direction of any triangle; choose it from the intended camera distance and final image resolution. The report includes per-material face assignments, UV ranges and repeat spans, image path, size, SHA-256, minimum and median density. A world transform and stage `metersPerUnit` are included in density. Out-of-range UVs require explicit `wrapS` or `wrapT` values of `repeat` or `mirror` on the corresponding axis. Missing/default wrap and `clamp` fail.

Scope: USD `Mesh` with valid vertex or faceVarying UVs, including indexed primvars, and direct `UsdPrimvarReader_float2` → `UsdUVTexture.st` → `UsdPreviewSurface.diffuseColor` connections. Per-face material subsets are checked independently. UDIM, UV transforms, other shader networks, missing image assets, malformed topology, collapsed UVs and degenerate or nonplanar polygons fail closed. The checker does not judge appearance, mip filtering, effective screen pixels, image blur, or whether an image's nominal resolution contains actual detail. Open a fixed-condition render at the intended observation distance for that judgment.
