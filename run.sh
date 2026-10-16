#!/usr/bin/env bash
# One-command reproduction: raw MCAP -> submission CSV (+ optional scoring).
#
# Modes:
#   baseline  Motion + ZUPT only (CPU, no map, robust fallback). Default.
#   full      Build ALIKED landmark maps (GPU) then PnP-fused localization.
#   validate  full, but on held-out validation laps with scoring.
#
# Usage:
#   ./run.sh <path/to/run.mcap> [mode]
#   MODE=full DEVICE=cuda CAMS="fl fr cl cr" ./run.sh run.mcap
#
# Env overrides (all optional):
#   MODE=baseline|full|validate        (default: baseline; or 1st-word of $2)
#   DEVICE=auto|cpu|cuda|cuda:N        (default: auto)
#   MATCHER=aliked_lightglue|sift      (default: aliked_lightglue)
#   CAMS="fl fr"                       cameras for the landmark map (default: fl)
#   OUT=outputs/submission.csv         submission path
#   STRIDE=1                           keyframe stride for map build (denser=1)
#   VAL_START / VAL_END                validation window seconds (validate mode)
#   EXCLUDE_START / EXCLUDE_END        laps to keep OUT of the map
#   GT=outputs/gt.csv                  ground-truth CSV (auto-extracted if tools found)
#   TOOLS=<dir>                        dir containing extract_gt.py / evaluate.py
#   REBUILD=1                          force rebuild of existing landmark maps
set -euo pipefail

# ---- locate repo + venv ----
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY="python"
if [[ -x "../.venv/bin/python" ]]; then PY="../.venv/bin/python"; fi
if [[ -x ".venv/bin/python" ]]; then PY=".venv/bin/python"; fi

# ---- args ----
MCAP="${1:-${MCAP:-}}"
MODE="${2:-${MODE:-baseline}}"
if [[ -z "${MCAP}" ]]; then
  echo "usage: ./run.sh <path/to/run.mcap> [baseline|full|validate]" >&2
  exit 2
fi
if [[ ! -f "${MCAP}" ]]; then
  echo "ERROR: mcap not found: ${MCAP}" >&2; exit 2
fi

DEVICE="${DEVICE:-auto}"
MATCHER="${MATCHER:-aliked_lightglue}"
CAMS="${CAMS:-fl}"
OUT="${OUT:-outputs/submission.csv}"
STRIDE="${STRIDE:-1}"
EXCLUDE_START="${EXCLUDE_START:-680}"
EXCLUDE_END="${EXCLUDE_END:-1063}"
VAL_START="${VAL_START:-680}"
VAL_END="${VAL_END:-833}"
MAP_DIR="${MAP_DIR:-outputs/maps}"
GT="${GT:-outputs/gt.csv}"

mkdir -p outputs "${MAP_DIR}"

# ---- locate starter-kit tools (extract_gt.py / evaluate.py) ----
find_tools() {
  if [[ -n "${TOOLS:-}" && -f "${TOOLS}/evaluate.py" ]]; then echo "${TOOLS}"; return; fi
  local d
  d="$(dirname "${MCAP}")"
  for cand in "${d}/tools" "${d}/../tools" "./tools"; do
    if [[ -f "${cand}/evaluate.py" ]]; then echo "${cand}"; return; fi
  done
  echo ""
}
TOOLS_DIR="$(find_tools)"

echo "=============================================================="
echo " camera-localization run.sh"
echo "   mcap    : ${MCAP}"
echo "   mode    : ${MODE}"
echo "   device  : ${DEVICE}    matcher: ${MATCHER}"
echo "   out     : ${OUT}"
echo "   tools   : ${TOOLS_DIR:-<not found; scoring skipped>}"
echo "=============================================================="

extra_run_args=()

build_maps() {
  local map_paths="{"
  local first=1
  for cam in ${CAMS}; do
    local mp="${MAP_DIR}/lm_${cam}.npz"
    if [[ "${REBUILD:-0}" == "1" || ! -f "${mp}" ]]; then
      echo ">>> [map] building landmark map for camera ${cam} ..."
      "${PY}" scripts/build_landmark_map.py \
        --mcap "${MCAP}" --cam "${cam}" --out "${mp}" \
        --matcher "${MATCHER}" --stride "${STRIDE}" \
        --exclude-start-s "${EXCLUDE_START}" --exclude-end-s "${EXCLUDE_END}" \
        --set runtime.device="${DEVICE}"
    else
      echo ">>> [map] reusing existing ${mp} (REBUILD=1 to force)"
    fi
    [[ ${first} -eq 1 ]] && first=0 || map_paths+=", "
    map_paths+="${cam}: ${mp}"
  done
  map_paths+="}"
  # cams list for pnp
  local cams_list; cams_list="[$(echo ${CAMS} | tr ' ' ',')]"
  extra_run_args=(
    --set pnp_reloc.enabled=true
    --set "pnp_reloc.map_paths=${map_paths}"
    --set "pnp_reloc.cams=${cams_list}"
    --set pnp_reloc.matcher="${MATCHER}"
    --set runtime.device="${DEVICE}"
  )
}

case "${MODE}" in
  baseline)
    echo ">>> [run] baseline (motion + ZUPT), CPU-friendly ..."
    "${PY}" run.py --mcap "${MCAP}" --out "${OUT}" --set runtime.device="${DEVICE}"
    ;;
  full)
    build_maps
    echo ">>> [run] PnP-fused localization over the full run ..."
    "${PY}" run.py --mcap "${MCAP}" --out "${OUT}" "${extra_run_args[@]}"
    ;;
  validate)
    build_maps
    echo ">>> [run] PnP-fused localization on validation window ${VAL_START}-${VAL_END}s ..."
    "${PY}" run.py --mcap "${MCAP}" --out "${OUT}" \
      --start-s "${VAL_START}" --end-s "${VAL_END}" "${extra_run_args[@]}"
    ;;
  *)
    echo "ERROR: unknown mode '${MODE}' (use baseline|full|validate)" >&2; exit 2;;
esac

echo ">>> [done] wrote ${OUT}"

# ---- optional scoring ----
if [[ -n "${TOOLS_DIR}" ]]; then
  if [[ ! -f "${GT}" ]]; then
    echo ">>> [gt] extracting ground truth -> ${GT}"
    "${PY}" "${TOOLS_DIR}/extract_gt.py" "${MCAP}" "${GT}"
  fi
  echo ">>> [score] ${TOOLS_DIR}/evaluate.py ${GT} ${OUT}"
  "${PY}" "${TOOLS_DIR}/evaluate.py" "${GT}" "${OUT}" || true
else
  echo ">>> [score] starter-kit tools not found; skipping scoring."
  echo "    set TOOLS=<dir with evaluate.py> to enable."
fi
