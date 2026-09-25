"""Recreate the small published-LangevinFlow overlay without changing the project."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REVISION = "68dcbfaa78b2371851842333ae933eea1f961832"
SOURCE_URL = "https://github.com/KingJamesSong/LangevinFlow_CCN.git"


def main() -> None:
    """Set up a version-pinned dependency overlay and verify model imports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    project = here.parents[2]
    source = here / "LangevinFlow_CCN"
    environment = here / "langevinflow-venv"
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required.")
    if not args.check_only:
        if not source.exists():
            subprocess.run(["git", "clone", SOURCE_URL, str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "switch", "--detach", REVISION], check=True)
        if not environment.exists():
            subprocess.run([uv, "venv", "--python", sys.executable, str(environment)], check=True)
        subprocess.run(
            [
                uv,
                "pip",
                "install",
                "--no-deps",
                "--cache-dir",
                "/tmp/churchland-lf-uv-cache",
                "--python",
                str(environment / "bin/python"),
                "-r",
                str(here / "langevinflow-requirements.txt"),
            ],
            check=True,
        )
        site = (
            environment
            / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
        )
        project_site = (
            project
            / f".venv/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
        )
        if not project_site.is_dir():
            raise RuntimeError(f"Project environment missing: {project_site}")
        (site / "churchland_project.pth").write_text(str(project_site) + "\n")
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise RuntimeError(f"Unexpected source revision {revision}; expected {REVISION}.")
    subprocess.run(
        [
            uv,
            "run",
            "--no-project",
            "--python",
            str(environment / "bin/python"),
            "python",
            str(here.parent / "scripts/langevinflow.py"),
            "--help",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
