"""Recreate the small pinned-ST-NDT overlay environment without changing the project."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

REVISION = "92ffe21c2042eb886389b7ebc97d300c2fd23cac"
SOURCE_URL = "https://github.com/trungle93/STNDT.git"
# The vendored copy carries no git metadata, so the pin is verified by content.
SOURCE_SHA256 = {
    "src/model.py": "08aa7d2157a849d863164ac199d3728b9c404fc34a836b42b3646c37560e455e",
    "src/mask.py": "8c8fbadc236a317a5624af431991788ce6e2043c825222b9a0364b5664f895b0",
}
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "STNDT"
DEFAULT_PROJECT_PYTHON = Path("/nfs/nhome/live/jbhagat/nldisco/.venv/bin/python")
DEFAULT_ENVIRONMENT = Path("/ceph/aeon/aeon/nldisco/rebuttal_20261118/change01_stndt/stndt-venv")


def source_digests() -> dict[str, str]:
    """Hash the two upstream modules whose code defines the model and its masking."""
    return {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
            for name in SOURCE_SHA256}


def verify_source() -> None:
    """Fail unless the vendored model and mask code equal the pinned revision."""
    actual = source_digests()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"ST-NDT source differs from revision {REVISION}: {actual}")


def main() -> None:
    """Set up a version-pinned dependency overlay and verify the pinned source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--print-digests", action="store_true")
    parser.add_argument("--project-python", type=Path, default=DEFAULT_PROJECT_PYTHON)
    parser.add_argument("--environment", type=Path, default=DEFAULT_ENVIRONMENT)
    args = parser.parse_args()
    if args.print_digests:
        print(source_digests())
        return
    verify_source()
    if args.check_only:
        return
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required.")
    version = subprocess.check_output(
        [str(args.project_python), "-c",
         "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        text=True).strip()
    if not args.environment.exists():
        subprocess.run([uv, "venv", "--python", str(args.project_python),
                        str(args.environment)], check=True)
    subprocess.run(
        [uv, "pip", "install", "--no-deps", "--cache-dir",
         str(args.environment.parent / "uv-cache"), "--python",
         str(args.environment / "bin/python"), "-r", str(HERE / "stndt-requirements.txt")],
        check=True)
    project_site = args.project_python.parents[1] / f"lib/python{version}/site-packages"
    if not project_site.is_dir():
        raise RuntimeError(f"Project environment missing: {project_site}")
    site = args.environment / f"lib/python{version}/site-packages"
    (site / "nldisco_project.pth").write_text(str(project_site) + "\n")
    subprocess.run([str(args.environment / "bin/python"), "-c",
                    "import torch, yacs"], check=True)
    print(f"ST-NDT overlay ready: {args.environment / 'bin/python'}", file=sys.stderr)


if __name__ == "__main__":
    main()
