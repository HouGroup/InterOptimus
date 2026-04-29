"""Download and verify InterOptimus MLIP checkpoints."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from InterOptimus.mlip import default_mlip_checkpoint_dir, resolve_mlip_checkpoint


@dataclass(frozen=True)
class CheckpointSpec:
    key: str
    calc: str
    filename: str
    url: str
    page_url: str

    @property
    def target_path(self) -> Path:
        return default_mlip_checkpoint_dir() / self.filename


CHECKPOINTS: tuple[CheckpointSpec, ...] = (
    CheckpointSpec(
        key="orb",
        calc="orb-models",
        filename="orb-v3-conservative-inf-mpa-20250404.ckpt",
        url=(
            "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/"
            "forcefields/orb-v3/orb-v3-conservative-inf-mpa-20250404.ckpt"
        ),
        page_url="https://matbench-discovery.materialsproject.org/models/orb-v3",
    ),
    CheckpointSpec(
        key="sevenn",
        calc="sevenn",
        filename="checkpoint_sevennet_omni_i12.pth",
        url="https://ndownloader.figshare.com/files/60977863",
        page_url="https://matbench-discovery.materialsproject.org/models/sevennet-omni-i12",
    ),
    CheckpointSpec(
        key="dpa",
        calc="dpa",
        filename="dpa-3.1-3m-ft.pth",
        url="https://ndownloader.figshare.com/files/55141895",
        page_url="https://matbench-discovery.materialsproject.org/models/dpa-3.1-3m-ft",
    ),
    CheckpointSpec(
        key="matris",
        calc="matris",
        filename="MatRIS_10M_OAM.pth.tar",
        url="https://ndownloader.figshare.com/files/59142728",
        page_url="https://matbench-discovery.materialsproject.org/models/matris-10m-oam",
    ),
)

_CHECKPOINT_BY_KEY = {spec.key: spec for spec in CHECKPOINTS}


def parse_checkpoint_selection(value: str | Iterable[str] | None) -> list[CheckpointSpec]:
    """Return checkpoint specs selected by comma-separated names or ``all``."""
    if value is None:
        names = ["all"]
    elif isinstance(value, str):
        names = [part.strip().lower() for part in value.split(",") if part.strip()]
    else:
        names = [str(part).strip().lower() for part in value if str(part).strip()]

    if not names or "all" in names:
        return list(CHECKPOINTS)

    unknown = [name for name in names if name not in _CHECKPOINT_BY_KEY]
    if unknown:
        valid = ", ".join(["all", *sorted(_CHECKPOINT_BY_KEY)])
        raise ValueError(f"unknown checkpoint name(s): {', '.join(unknown)}. Valid values: {valid}")
    return [_CHECKPOINT_BY_KEY[name] for name in names]


def _format_size(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown size"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{num_bytes} B"


def checkpoint_status(spec: CheckpointSpec) -> tuple[bool, Path | None]:
    """Return whether a checkpoint is present and the path that satisfied it."""
    path = spec.target_path
    if path.is_file() and path.stat().st_size > 0:
        return True, path

    # Report an alternate match for context, but keep the status missing because
    # the managed setup expects the specific model version in the manifest.
    if spec.calc in {"orb-models", "sevenn", "dpa"}:
        resolved = resolve_mlip_checkpoint(spec.calc)
        return False, Path(resolved) if resolved else None
    return False, None


def print_manual_download_help(spec: CheckpointSpec) -> None:
    target = spec.target_path
    print(f"  {spec.key}: 可手动下载后放到:")
    print(f"    {target}")
    print(f"    direct: {spec.url}")
    print(f"    page:   {spec.page_url}")


def download_checkpoint(spec: CheckpointSpec, *, force: bool = False, timeout: int = 60) -> Path:
    """Download one checkpoint to the InterOptimus checkpoint cache."""
    target = spec.target_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0 and not force:
        print(f"{spec.key}: 已存在，跳过: {target}")
        return target

    request = urllib.request.Request(spec.url, headers={"User-Agent": "InterOptimus/checkpoints"})
    tmp_name: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            total = response.headers.get("Content-Length")
            total_bytes = int(total) if total and total.isdigit() else None
            print(f"{spec.key}: 下载 {_format_size(total_bytes)} -> {target}")
            with tempfile.NamedTemporaryFile(
                "wb",
                dir=str(target.parent),
                prefix=f".{target.name}.",
                suffix=".part",
                delete=False,
            ) as tmp:
                tmp_name = tmp.name
                shutil.copyfileobj(response, tmp)
        tmp_path = Path(tmp_name)
        if tmp_path.stat().st_size <= 0:
            raise RuntimeError("downloaded file is empty")
        tmp_path.replace(target)
        print(f"{spec.key}: 下载完成: {target}")
        return target
    except (OSError, urllib.error.URLError, RuntimeError) as exc:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)
        raise RuntimeError(f"{spec.key}: download failed from {spec.url}: {exc}") from exc


def download_checkpoints(
    specs: Iterable[CheckpointSpec],
    *,
    force: bool = False,
    timeout: int = 60,
) -> bool:
    """Download selected checkpoints. Returns True when all are present afterward."""
    ok = True
    for spec in specs:
        try:
            download_checkpoint(spec, force=force, timeout=timeout)
        except RuntimeError as exc:
            ok = False
            print(f"警告: {exc}", file=sys.stderr)
            print_manual_download_help(spec)
    return verify_checkpoints(specs)


def verify_checkpoints(specs: Iterable[CheckpointSpec]) -> bool:
    """Print checkpoint readiness and return True if all selected checkpoints exist."""
    all_ok = True
    print(f"Checkpoint 目录: {default_mlip_checkpoint_dir()}")
    for spec in specs:
        ok, path = checkpoint_status(spec)
        if ok and path is not None:
            print(f"  OK      {spec.key}: {path}")
        else:
            all_ok = False
            print(f"  MISSING {spec.key}: expected {spec.target_path}")
            if path is not None:
                print(f"          found alternate checkpoint: {path}")
            print_manual_download_help(spec)
    return all_ok


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download and verify InterOptimus MLIP checkpoints.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List known checkpoint URLs and target paths")
    p_list.add_argument("models", nargs="?", default="all", help="all or comma-separated: orb,sevenn,dpa,matris")

    p_download = sub.add_parser("download", help="Download checkpoints to the InterOptimus cache")
    p_download.add_argument("models", nargs="?", default="all", help="all or comma-separated: orb,sevenn,dpa,matris")
    p_download.add_argument("--force", action="store_true", help="re-download even if a local file exists")
    p_download.add_argument("--timeout", type=int, default=60, help="per-request timeout in seconds")

    p_verify = sub.add_parser("verify", help="Verify whether checkpoints are present")
    p_verify.add_argument("models", nargs="?", default="all", help="all or comma-separated: orb,sevenn,dpa,matris")

    args = parser.parse_args(argv)
    try:
        specs = parse_checkpoint_selection(args.models)
    except ValueError as exc:
        parser.error(str(exc))

    if args.command == "list":
        print(f"Checkpoint 目录: {default_mlip_checkpoint_dir()}")
        for spec in specs:
            print(f"{spec.key}:")
            print(f"  target: {spec.target_path}")
            print(f"  direct: {spec.url}")
            print(f"  page:   {spec.page_url}")
        return

    if args.command == "download":
        if not download_checkpoints(specs, force=args.force, timeout=args.timeout):
            raise SystemExit(1)
        return

    if args.command == "verify":
        if not verify_checkpoints(specs):
            raise SystemExit(1)
        return


if __name__ == "__main__":
    main()
