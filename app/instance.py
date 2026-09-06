"""ERA AI — portable instance (spec Section 27).

All user state lives under the instance/ folder. This module zips it
into a single <name>.era file for copy-paste portability, and restores
it on another machine. This is the true "run anywhere" layer.
"""
from __future__ import annotations

import shutil
import time
import zipfile
from pathlib import Path


def export_instance(instance_dir: Path, out_path: Path) -> Path:
    """Zip the instance folder into out_path (.era)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ERA_INSTANCE", f"exported={time.time()}\n")
        if instance_dir.exists():
            for f in instance_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(instance_dir.parent))
    return out_path


def import_instance(era_path: Path, target_root: Path,
                   overwrite: bool = False) -> Path:
    """Restore an .era archive into target_root/instance."""
    with zipfile.ZipFile(era_path, "r") as zf:
        names = zf.namelist()
        if "ERA_INSTANCE" not in names:
            raise ValueError("Not a valid .era instance file")
        target_instance = target_root / "instance"
        if target_instance.exists():
            if not overwrite:
                raise FileExistsError(
                    "instance/ already exists; pass overwrite=True to replace")
            shutil.rmtree(target_instance)
        for n in names:
            if n == "ERA_INSTANCE":
                continue
            zf.extract(n, target_root)
    return target_root / "instance"
