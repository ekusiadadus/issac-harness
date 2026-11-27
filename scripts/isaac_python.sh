#!/usr/bin/env bash
# Run an Isaac script with host paths preserved across the container boundary.
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
out=$(realpath "${HARNESS_OUTPUT_ROOT:?Set HARNESS_OUTPUT_ROOT to an existing writable output directory}")
kit_cache=${HARNESS_KIT_CACHE:-$out/.kit-cache}
mkdir -p "$kit_cache"
mode=${HARNESS_RUNTIME:-sim}
case "$mode" in
  sim) image=nvcr.io/nvidia/isaac-sim@sha256:a025b65a7442d56806a1cc3f5eed081caf736c24b72d62dccf68ef66945d8690
       launcher=(--entrypoint /isaac-sim/python.sh "$image") ;;
  lab) image=nvcr.io/nvidia/isaac-lab@sha256:f70acc9ed3bc3fc67099a0c5715379c432cc9965623cabbaf54f74599997e245
       launcher=(--entrypoint /bin/bash "$image" -lc 'cd /workspace/isaaclab && exec ./isaaclab.sh -p "$@"' _) ;;
  *) echo 'HARNESS_RUNTIME must be sim or lab' >&2; exit 2 ;;
esac
engine=(docker)
if ! docker info >/dev/null 2>&1; then engine=(sudo -n docker); fi
extra=()
if [[ -r /usr/share/nvidia/nvoptix.bin ]]; then extra+=(-v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro); fi
if [[ -n ${HARNESS_PROJECT_ROOT:-} ]]; then
  project_root=$(realpath "$HARNESS_PROJECT_ROOT")
  if [[ "$project_root" != "$root" && "$project_root" != "$out" ]]; then
    extra+=(-v "$project_root:$project_root:ro")
  fi
fi
if [[ "$mode" == sim ]]; then
  mkdir -p "$kit_cache/nv_shadercache"
  extra+=(-v "$kit_cache/nv_shadercache:/isaac-sim/kit/cache/nv_shadercache")
fi
exec "${engine[@]}" run --rm --gpus all --shm-size=4g --user 0:0 \
  -e "ACCEPT_EULA=${ACCEPT_EULA:?Set ACCEPT_EULA for the NVIDIA runtime}" -e PRIVACY_CONSENT=N \
  -e OMNI_KIT_ALLOW_ROOT=1 -e OMNICLIENT_HUB_MODE=disabled -e PYTHONUNBUFFERED=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute,display,video \
  -v "$root:$root:ro" -v "$out:$out:rw" -v "$kit_cache:/root/.cache/ov" \
  "${extra[@]}" "${launcher[@]}" "$@"
