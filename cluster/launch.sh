#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON=/opt/red-patterns/.venv/bin/python
DIAGNOSTICS=runtime-diagnostics.txt

if ! IMAGE_TAG="${IMAGE_TAG:-<unset>}" "$PYTHON" - <<'PY' >"$DIAGNOSTICS" 2>&1
import importlib
import importlib.metadata
import os
import pathlib
import sys

print(f"requested_image_tag={os.environ['IMAGE_TAG']}")
revision_path = pathlib.Path("/opt/red-patterns/image-revision")
if revision_path.is_file():
    print(f"embedded_image_revision={revision_path.read_text().strip()}")
else:
    print("embedded_image_revision=<missing>")
print(f"python_executable={sys.executable}")
print(f"python_version={sys.version}")

missing = []
for name in ("h5py", "numpy", "pydantic"):
    try:
        module = importlib.import_module(name)
        print(f"{name}_version={importlib.metadata.version(name)}")
        print(f"{name}_path={module.__file__}")
    except Exception as error:
        missing.append(name)
        print(f"{name}_import_error={error!r}")

if missing:
    print(f"python_path={sys.path!r}")
    raise SystemExit(f"Missing required sweep dependencies: {', '.join(missing)}")
PY
then
	printf 'Sweep runtime preflight failed; see %s for the requested image tag, embedded revision, and Python import errors.\n' "$DIAGNOSTICS" >&2
	exit 1
fi

exec "$PYTHON" /opt/red-patterns/sweep/run_one.py --runs-jsonl runs.jsonl --run-id "$1"
