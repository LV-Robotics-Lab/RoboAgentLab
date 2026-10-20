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

    python -m pi_embodied_services.robots.libero.assets [--workers N]
        fetch RLinf/LIBERO-PRO-assets at a pinned revision into the Hugging Face cache, check
        every file the Hub lists for that revision, then link (the tree LIBERO-PRO uses; it
        holds standard LIBERO's scenes)
    python -m pi_embodied_services.robots.libero.assets --from-package liberopro.liberopro
        link to the tree another installed package already uses (setup.sh libero-pro, after
        LIBERO-PRO's own verified download)

A directory that already holds the probe scene is kept. A partial download is never linked:
``snapshot_download`` returns whatever snapshot is cached when the Hub cannot be reached,
complete or not (the defect liberopro-verify-assets.patch fixes in LIBERO-PRO's downloader), so
the files are listed and fetched one by one here. A rerun resumes: files already in the cache
are not fetched again. Mirrors rate-limit (hf-mirror.com answers 429 to eight parallel
downloads), so with HF_ENDPOINT on a mirror two files are fetched at a time, and every Hub
request is retried with backoff on a 429 or 5xx answer. Honours HF_ENDPOINT and HF_TOKEN
(huggingface_hub reads them).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Optional, TypeVar
from urllib.parse import urlparse

REPO = "RLinf/LIBERO-PRO-assets"
REVISION = "e4b10c9adc5735c9c8776917527178e693ef10b3"
#: A scene the first libero_10 task loads; check.ts's preflight probes the same file.
PROBE = "scenes/libero_living_room_tabletop_base_style.xml"
#: Parallel downloads from huggingface.co and from a mirror (HF_ENDPOINT).
HUB_WORKERS, MIRROR_WORKERS = 8, 2
#: Waits (s) before each retry of a request answered 429 or 5xx; a shorter Retry-After wins.
BACKOFF = (10, 20, 40, 80, 120)

T = TypeVar("T")


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


def _chain(e: BaseException) -> Iterable[BaseException]:
    """``e`` and the exceptions it was raised from (huggingface_hub wraps a failed metadata
    request in LocalEntryNotFoundError, so the HTTP response sits further down)."""
    seen: set[int] = set()
    cur: Optional[BaseException] = e
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


def http_status(e: BaseException) -> Optional[int]:
    for x in _chain(e):
        code = getattr(getattr(x, "response", None), "status_code", None)
        if isinstance(code, int):
            return code
    return None


def _retry_after(e: BaseException) -> Optional[float]:
    for x in _chain(e):
        headers = getattr(getattr(x, "response", None), "headers", None) or {}
        try:
            return float(headers.get("Retry-After"))
        except (TypeError, ValueError):
            continue
    return None


def retrying(
    what: str,
    call: Callable[[], T],
    sleep: Callable[[float], None] = time.sleep,
    delays: tuple[float, ...] = BACKOFF,
) -> T:
    """``call()``, retried after each wait in ``delays`` while it fails with a 429 or 5xx answer."""
    for attempt in range(len(delays) + 1):
        try:
            return call()
        except Exception as e:
            status = http_status(e)
            if (
                attempt == len(delays)
                or status is None
                or not (status == 429 or status >= 500)
            ):
                raise
            hinted = _retry_after(e)
            wait = min(hinted, delays[attempt]) if hinted else delays[attempt]
            print(
                f"LIBERO assets: {what}: HTTP {status}, retry {attempt + 1}/{len(delays)} in {wait:g} s",
                file=sys.stderr,
            )
            sleep(wait)
    raise AssertionError("unreachable")


def mirror() -> Optional[str]:
    """HF_ENDPOINT when it points somewhere other than huggingface.co."""
    endpoint = os.environ.get("HF_ENDPOINT", "").strip()
    host = urlparse(endpoint).hostname or ""
    return endpoint if host and host != "huggingface.co" else None


def default_workers() -> int:
    return MIRROR_WORKERS if mirror() else HUB_WORKERS


class Hub:
    """The Hub calls ``fetch`` makes, through huggingface_hub (tests pass a fake)."""

    def files(self, repo: str, revision: str) -> list[str]:
        from huggingface_hub import HfApi

        # repo_info: one request (the paginated tree API is not served by every mirror).
        info = HfApi().repo_info(repo_id=repo, repo_type="dataset", revision=revision)
        return [s.rfilename for s in info.siblings or []]

    def snapshot(self, repo: str, revision: str) -> Path:
        """Where the cache keeps ``repo`` at the commit ``revision``."""
        from huggingface_hub import constants
        from huggingface_hub.file_download import repo_folder_name

        folder = repo_folder_name(repo_id=repo, repo_type="dataset")
        return Path(constants.HF_HUB_CACHE) / folder / "snapshots" / revision

    def download(self, repo: str, revision: str, filename: str) -> None:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=repo, repo_type="dataset", revision=revision, filename=filename
        )


def _hint() -> str:
    m = mirror()
    if not m:
        return "rerun to resume"
    return (
        f"rerun to resume, or unset HF_ENDPOINT (now {m}) to fetch from huggingface.co "
        "(through a proxy where it is unreachable)"
    )


def fetch(
    repo: str = REPO,
    revision: str = REVISION,
    hub: Optional[Hub] = None,
    workers: Optional[int] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """The verified snapshot of ``repo`` at the commit ``revision`` in the Hugging Face cache.

    Lists the files at ``revision``, fetches the ones the cache lacks (``workers`` at a time),
    and checks that every listed file is there. The first file that still fails after its
    retries stops the fetch; the files already fetched stay for the next run.
    """
    hub = hub or Hub()
    tag = f"{repo}@{revision[:8]}"
    root = hub.snapshot(repo, revision)
    try:
        expected = [
            f
            for f in retrying(f"list {tag}", lambda: hub.files(repo, revision), sleep)
            if not f.startswith(".git")
        ]
    except Exception as e:  # noqa: BLE001 - cannot verify without the file list
        raise RuntimeError(
            f"could not list {tag} to verify {root}: {e}; {_hint()}"
        ) from e
    todo = missing_files(root, expected)
    first_error: Optional[tuple[str, BaseException]] = None
    if todo:
        n = max(1, workers or default_workers())
        print(
            f"LIBERO assets: fetching {len(todo)} of {len(expected)} files of {tag}"
            f" into {root}, {n} at a time",
            file=sys.stderr,
        )
        stop = threading.Event()  # set by the first file that fails for good

        def one(f: str) -> None:
            if stop.is_set():
                return
            try:
                retrying(f, lambda: hub.download(repo, revision, f), sleep)
            except Exception:
                stop.set()
                raise

        with ThreadPoolExecutor(max_workers=n) as pool:
            jobs = {pool.submit(one, f): f for f in todo}
            for job in as_completed(jobs):
                err = job.exception()
                if err is not None and first_error is None:
                    first_error = (jobs[job], err)
    gone = missing_files(root, expected)
    if gone:
        why = f"; {first_error[0]}: {first_error[1]}" if first_error else ""
        raise RuntimeError(
            f"incomplete download of {tag}: {len(gone)} of {len(expected)} files missing "
            f"under {root} (e.g. {gone[:3]}){why}; the {len(expected) - len(gone)} files "
            f"already fetched are kept: {_hint()}"
        )
    if not present(root):
        raise RuntimeError(f"{tag} has no {PROBE}")
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
        "--workers",
        type=int,
        help=f"parallel downloads (default: {MIRROR_WORKERS} with HF_ENDPOINT on a mirror, "
        f"else {HUB_WORKERS})",
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
            target = fetch(workers=args.workers)
        link(assets, target)
    except RuntimeError as e:
        print(f"LIBERO assets: {e}", file=sys.stderr)
        return 1
    print(f"LIBERO assets at {assets} -> {os.fspath(target)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
