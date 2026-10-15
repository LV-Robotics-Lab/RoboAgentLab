# Copyright 2026 The pi-embodied Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standard LIBERO's scene assets (services/setup.sh libero, libero-pro).

The ``[libero]`` install ships no ``libero/libero/assets``, the directory LIBERO's env code
reads every scene and mesh from, so the first ``env.reset`` of a fresh install dies on a
missing scene file. This links that directory to a complete assets tree:

    python -m pi_embodied_services.robots.libero.assets
        fetch RLinf/LIBERO-PRO-assets at a pinned revision into the Hugging Face cache, check
        every file the Hub lists for that revision, then link (the tree LIBERO-PRO uses; it
        holds standard LIBERO's scenes)
    python -m pi_embodied_services.robots.libero.assets --from-package liberopro.liberopro
        link to the tree another installed package already uses (setup.sh libero-pro, after
        LIBERO-PRO's own verified download)

A directory that already holds the probe scene is kept. A partial download is never linked:
``snapshot_download`` returns whatever snapshot is cached when the Hub cannot be reached,
complete or not (the defect liberopro-verify-assets.patch fixes in LIBERO-PRO's downloader).
Honours HF_ENDPOINT and HF_TOKEN (huggingface_hub reads them).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional

REPO = "RLinf/LIBERO-PRO-assets"
REVISION = "e4b10c9adc5735c9c8776917527178e693ef10b3"
#: A scene the first libero_10 task loads; check.ts's preflight probes the same file.
PROBE = "scenes/libero_living_room_tabletop_base_style.xml"


def package_assets_dir(module: str = "libero.libero") -> Path:
    """``<package dir>/assets`` of an installed package, without importing it."""
    try:
        spec = importlib.util.find_spec(module)
    except ModuleNotFoundError:
        spec = None
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(f"{module} is not installed in {sys.executable}")
    return Path(list(spec.submodule_search_locations)[0]) / "assets"


def present(assets: Path) -> bool:
    return (assets / PROBE).is_file()


def missing_files(root: Path, expected: Iterable[str]) -> list[str]:
    return [f for f in expected if not f.startswith(".git") and not (root / f).exists()]


def fetch(
    repo: str = REPO,
    revision: str = REVISION,
    download: Optional[Callable[..., str]] = None,
    list_files: Optional[Callable[[str, str], list[str]]] = None,
) -> Path:
    """The verified snapshot of ``repo`` at ``revision`` in the Hugging Face cache."""
    if download is None or list_files is None:
        from huggingface_hub import HfApi, snapshot_download

        def hub_files(r: str, rev: str) -> list[str]:
            # repo_info: one request (the paginated tree API is not served by every mirror).
            info = HfApi().repo_info(repo_id=r, repo_type="dataset", revision=rev)
            return [s.rfilename for s in info.siblings or []]

        download = download or snapshot_download
        list_files = list_files or hub_files
    root = Path(download(repo_id=repo, repo_type="dataset", revision=revision))
    try:
        expected = list_files(repo, revision)
    except Exception as e:  # noqa: BLE001 - cannot verify without the file list
        raise RuntimeError(
            f"could not list {repo}@{revision[:8]} to verify {root}: {e}"
        ) from e
    gone = missing_files(root, expected)
    if gone:
        raise RuntimeError(
            f"incomplete download of {repo}: {len(gone)} of {len(expected)} files missing "
            f"under {root} (e.g. {gone[:3]}); rerun to resume"
        )
    if not present(root):
        raise RuntimeError(f"{repo}@{revision[:8]} has no {PROBE}")
    return root


def link(assets: Path, target: Path) -> None:
    """Make ``assets`` a symlink to ``target``; an empty directory or an old link is replaced."""
    if assets.is_symlink():
        assets.unlink()
    elif assets.is_dir():
        if any(assets.iterdir()):
            raise RuntimeError(
                f"{assets} holds files but not {PROBE}: move it away and rerun"
            )
        assets.rmdir()
    assets.symlink_to(target, target_is_directory=True)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--from-package",
        metavar="MODULE",
        help="link to the assets tree MODULE uses (e.g. liberopro.liberopro)",
    )
    p.add_argument(
        "--assets-dir",
        type=Path,
        help="the directory to link (default: the installed libero.libero's assets)",
    )
    args = p.parse_args(argv)
    try:
        assets = args.assets_dir or package_assets_dir()
        if present(assets):
            print(f"LIBERO assets present at {assets}")
            return 0
        if args.from_package:
            target = package_assets_dir(args.from_package).resolve()
            if not present(target):
                raise RuntimeError(f"{args.from_package} has no assets at {target}")
        else:
            target = fetch()
        link(assets, target)
    except RuntimeError as e:
        print(f"LIBERO assets: {e}", file=sys.stderr)
        return 1
    print(f"LIBERO assets at {assets} -> {os.fspath(target)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
