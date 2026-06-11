#!/usr/bin/env bash
# run.sh — launch Streamlit for the Home Assistant add-on.
#
# HA serves add-ons through its ingress proxy under a dynamic base path like
# /api/hassio_ingress/<token>/ . Streamlit needs that path via
# --server.baseUrlPath or the sidebar panel renders blank. We ask the
# Supervisor API for this add-on's ingress entry; if that's unavailable we
# fall back to root (works for direct LAN access on :8501).
set -e

BASE_PATH=""
if [ -n "${SUPERVISOR_TOKEN:-}" ]; then
  BASE_PATH="$(
    curl -s -H "Authorization: Bearer ${SUPERVISOR_TOKEN}" \
      http://supervisor/addons/self/info \
      | sed -n 's/.*"ingress_entry": *"\([^"]*\)".*/\1/p'
  )" || true
fi

ARGS=(
  --server.address=0.0.0.0
  --server.port=8501
  --server.headless=true
  --server.enableCORS=false
  --server.enableXsrfProtection=false
  --browser.gatherUsageStats=false
)

if [ -n "$BASE_PATH" ]; then
  ARGS+=( --server.baseUrlPath="${BASE_PATH#/}" )
fi

echo "Starting Spectra-Lab (baseUrlPath='${BASE_PATH:-/}')"
exec streamlit run /app/app.py "${ARGS[@]}"
