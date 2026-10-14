#!/usr/bin/env bash
# awscloudscan launcher — runs the tool from this directory without installation.
# Usage: ./awscloudscan.sh -p <profile-name> -r <region>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${HERE}/run.py" "$@"
