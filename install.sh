#!/usr/bin/env bash
# Install awscloudscan as an `awscloudscan` command on your PATH.
#
# Why a symlink and not `pip install -e .`: an editable install records the
# absolute source path at install time — a PEP 660 finder module in site-packages
# plus a `direct_url.json` — so moving or renaming the checkout leaves an
# `awscloudscan` command that dies with `ModuleNotFoundError: No module named
# 'cloudscan'` no matter where you run it from. A symlink records nothing: it
# resolves to wherever the checkout lives now, and if the checkout is moved it
# fails honestly (`No such file or directory`) instead of with a traceback from
# pip's finder. Re-run this script after moving the checkout, or after a `git
# pull` that adds dependencies.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BIN_DIR="${HOME}/.local/bin"
LAUNCHER="${BIN_DIR}/awscloudscan"

echo "[*] installing awscloudscan from ${HERE}"

# 1. Sanity-check the checkout before touching anything on PATH, so a broken
#    source tree is reported here rather than by a launcher that looks installed.
if ! python3 "${HERE}/run.py" --version >/dev/null; then
    echo "[!] ${HERE}/run.py does not run — fix that first." >&2
    exit 1
fi

# 2. Remove a stale editable install. Only an *editable* install is a candidate:
#    that marker means "installed from a local source directory", so it can never
#    be an unrelated PyPI package that happens to be called cloudscan or
#    awscloudscan (the old and new distribution names for this checkout). A
#    normal install is left alone; our symlink simply takes its place on PATH.
#
#    Two traps this code steps around. The checkout's own `cloudscan.egg-info`
#    sits in the current directory, and the current directory is on sys.path for
#    a script read from stdin — so asking importlib for "the" cloudscan
#    distribution from the repo root returns that egg-info, which has no
#    direct_url.json, and the stale install goes undetected. Hence: scan every
#    distribution under either name and take the one that is actually editable.
#    For the same reason the pip calls run from /tmp, so pip cannot resolve the
#    uninstall to the egg-info instead of the installed distribution.
EDITABLE_DIST="$(python3 - <<'PY'
import json
from importlib import metadata

for dist in metadata.distributions():
    name = (dist.metadata["Name"] or "").lower()
    if name not in ("cloudscan", "awscloudscan"):
        continue
    try:
        raw = dist.read_text("direct_url.json")
    except OSError:
        continue
    if not raw:
        continue
    try:
        data = json.loads(raw)
    except ValueError:
        continue
    if data.get("dir_info", {}).get("editable"):
        src = data.get("url", "").removeprefix("file://")
        print(f"{name}\t{src}")
        break
PY
)"

if [[ -n "${EDITABLE_DIST}" ]]; then
    DIST_NAME="${EDITABLE_DIST%%$'\t'*}"
    EDITABLE_SRC="${EDITABLE_DIST#*$'\t'}"
    echo "[*] removing editable install of ${DIST_NAME} (source recorded as ${EDITABLE_SRC})"
    if ! ( cd /tmp && python3 -m pip uninstall -y "${DIST_NAME}" ) >/dev/null 2>&1; then
        ( cd /tmp && python3 -m pip uninstall -y --break-system-packages "${DIST_NAME}" )
    fi
fi

# 3. Install the launcher symlink.
mkdir -p "${BIN_DIR}"
chmod +x "${HERE}/run.py"
ln -sfn "${HERE}/run.py" "${LAUNCHER}"

# 4. Retire the old launcher. A rename should leave exactly one command behind,
#    so a leftover `cloudscan` symlink from a previous install is removed.
if [[ -L "${BIN_DIR}/cloudscan" ]]; then
    OLD_TARGET="$(readlink "${BIN_DIR}/cloudscan")"
    rm -f "${BIN_DIR}/cloudscan"
    echo "[*] removed the old cloudscan launcher (-> ${OLD_TARGET})"
fi

# 5. Verify from a directory that has nothing to do with the checkout — the exact
#    case that used to fail. Better to fail at install time than at scan time.
if ( cd /tmp && "${LAUNCHER}" --version >/dev/null ); then
    echo "[+] installed: ${LAUNCHER} -> ${HERE}/run.py"
else
    echo "[!] ${LAUNCHER} is installed but does not run from another directory." >&2
    exit 1
fi

if ! command -v awscloudscan >/dev/null; then
    echo "[+] add ~/.local/bin to your PATH:"
    echo "    export PATH=\"\$PATH:\$HOME/.local/bin\""
fi
