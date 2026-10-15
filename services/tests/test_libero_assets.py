"""setup.sh libero's scene assets (robots/libero/assets.py): a partial snapshot is never
linked, a complete one is, a rerun resumes, 429/5xx answers are retried with backoff, a mirror
gets fewer parallel downloads, an existing tree is kept, and libero-pro links LIBERO-PRO's tree."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from pi_embodied_services.robots.libero import assets as A

FILES = [".gitattributes", A.PROBE, "textures/wood.png"]


def snapshot(root: Path, files: list[str]) -> Path:
    for f in files:
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text("x")
    return root


class HTTPError(Exception):
    """What huggingface_hub raises for an HTTP answer: the response is on ``.response``."""

    def __init__(self, status: int, retry_after: str | None = None):
        super().__init__(f"{status} Client Error")
        headers = {"Retry-After": retry_after} if retry_after else {}
        self.response = SimpleNamespace(status_code=status, headers=headers)


class FakeHub:
    """A Hub whose answers are scripted: ``fail[name]`` lists the errors the first calls for
    that file (or ``LIST`` for the listing) raise before the call succeeds."""

    LIST = "<list>"

    def __init__(self, root: Path, listed=FILES, fail=None):
        self.root, self.listed = root, list(listed)
        self.fail = {k: list(v) for k, v in (fail or {}).items()}
        self.calls: list[str] = []
        self.lock = threading.Lock()

    def _answer(self, name: str) -> None:
        with self.lock:
            self.calls.append(name)
            errors = self.fail.get(name)
            if errors:
                raise errors.pop(0)

    def files(self, repo, revision):
        assert (repo, revision) == (A.REPO, A.REVISION)
        self._answer(self.LIST)
        return list(self.listed)

    def snapshot(self, repo, revision):
        return self.root

    def download(self, repo, revision, filename):
        assert (repo, revision) == (A.REPO, A.REVISION)
        self._answer(filename)
        snapshot(self.root, [filename])


def fetch(hub, **kw):
    waits: list[float] = []
    kw.setdefault("workers", 1)
    return A.fetch(hub=hub, sleep=waits.append, **kw), waits


def test_fetch_lists_the_pinned_revision_and_fetches_every_file(tmp_path):
    hub = FakeHub(tmp_path / "snap")
    root, waits = fetch(hub)
    assert root == tmp_path / "snap"
    assert hub.calls == [FakeHub.LIST, *FILES[1:]]  # .gitattributes is not checked
    assert waits == []


def test_a_rerun_fetches_only_what_the_cache_lacks(tmp_path):
    hub = FakeHub(snapshot(tmp_path / "snap", [A.PROBE]))
    fetch(hub)
    assert hub.calls == [FakeHub.LIST, "textures/wood.png"]
    hub.calls.clear()
    fetch(hub)  # complete: only the listing that verifies it
    assert hub.calls == [FakeHub.LIST]


def test_429_and_5xx_answers_are_retried_with_backoff(tmp_path):
    hub = FakeHub(
        tmp_path / "snap",
        fail={
            FakeHub.LIST: [HTTPError(429)],
            A.PROBE: [HTTPError(429, retry_after="3"), HTTPError(503)],
        },
    )
    _, waits = fetch(hub)
    assert waits == [A.BACKOFF[0], 3, A.BACKOFF[1]]
    assert hub.calls.count(FakeHub.LIST) == 2 and hub.calls.count(A.PROBE) == 3
    assert A.present(tmp_path / "snap")


def test_a_429_wrapped_by_huggingface_hub_is_retried(tmp_path):
    """hf_hub_download turns a failed metadata request into LocalEntryNotFoundError."""

    def wrapped():
        try:
            raise HTTPError(429)
        except HTTPError as e:
            try:
                raise FileNotFoundError("cannot find the requested files") from e
            except FileNotFoundError as w:
                return w

    hub = FakeHub(tmp_path / "snap", fail={"textures/wood.png": [wrapped()]})
    _, waits = fetch(hub)
    assert waits == [A.BACKOFF[0]]


def test_other_errors_are_not_retried(tmp_path):
    hub = FakeHub(tmp_path / "snap", fail={A.PROBE: [HTTPError(404)]})
    with pytest.raises(RuntimeError, match=r"2 of 2 files missing.*404 Client Error"):
        fetch(hub)
    assert hub.calls.count(A.PROBE) == 1


def test_a_file_that_keeps_failing_stops_the_fetch_and_keeps_the_rest(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    listed = [f"f{i}" for i in range(6)]
    hub = FakeHub(
        tmp_path / "snap",
        listed=listed,
        fail={"f1": [HTTPError(429)] * (len(A.BACKOFF) + 1)},
    )
    with pytest.raises(RuntimeError) as e:
        fetch(hub)
    msg = str(e.value)
    assert "f1: 429 Client Error" in msg
    assert "rerun to resume, or unset HF_ENDPOINT (now https://hf-mirror.com)" in msg
    # One worker: f0 was fetched, f1 used up its retries, the files after it were skipped.
    assert hub.calls == [FakeHub.LIST, "f0", *["f1"] * (len(A.BACKOFF) + 1)]
    assert (tmp_path / "snap" / "f0").exists()
    # The next run resumes at f1.
    hub.calls.clear()
    with pytest.raises(
        RuntimeError, match="has no"
    ):  # complete, but this fake has no PROBE
        fetch(hub)
    assert hub.calls == [FakeHub.LIST, *listed[1:]]


def test_an_unlistable_hub_is_refused_not_trusted(tmp_path):
    """The offline fallback: the cache holds a snapshot, but nothing can verify it."""
    hub = FakeHub(
        snapshot(tmp_path / "snap", FILES[1:]),
        fail={FakeHub.LIST: [OSError("hub unreachable")]},
    )
    with pytest.raises(RuntimeError, match="could not list.*hub unreachable"):
        fetch(hub)


def test_a_listing_that_keeps_answering_429_is_refused(tmp_path):
    hub = FakeHub(
        tmp_path / "snap", fail={FakeHub.LIST: [HTTPError(429)] * (len(A.BACKOFF) + 1)}
    )
    with pytest.raises(RuntimeError, match="could not list.*429"):
        fetch(hub)
    assert hub.calls == [FakeHub.LIST] * (len(A.BACKOFF) + 1)


@pytest.mark.parametrize(
    "endpoint, workers",
    [
        (None, A.HUB_WORKERS),
        ("https://huggingface.co", A.HUB_WORKERS),
        ("https://hf-mirror.com", A.MIRROR_WORKERS),
        ("https://hf-mirror.com/", A.MIRROR_WORKERS),
    ],
)
def test_a_mirror_gets_fewer_parallel_downloads(monkeypatch, endpoint, workers):
    if endpoint is None:
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
    else:
        monkeypatch.setenv("HF_ENDPOINT", endpoint)
    assert A.default_workers() == workers


def test_parallel_workers_fetch_everything(tmp_path):
    listed = [A.PROBE, *[f"m/{i}.obj" for i in range(40)]]
    hub = FakeHub(tmp_path / "snap", listed=listed, fail={"m/7.obj": [HTTPError(502)]})
    root, _ = fetch(hub, workers=4)
    assert A.missing_files(root, listed) == []


def test_main_links_the_package_dir_and_keeps_a_present_tree(tmp_path, monkeypatch):
    root = snapshot(tmp_path / "snap", FILES[1:])
    monkeypatch.setattr(A, "fetch", lambda **kw: root)
    pkg = tmp_path / "libero" / "libero"
    pkg.mkdir(parents=True)
    (pkg / "assets").mkdir()  # an empty directory, as a wheel without assets leaves
    assert A.main(["--assets-dir", str(pkg / "assets")]) == 0
    assert (pkg / "assets").resolve() == root.resolve()
    assert A.present(pkg / "assets")
    # A second run keeps it without fetching.
    monkeypatch.setattr(A, "fetch", lambda **kw: pytest.fail("fetched again"))
    assert A.main(["--assets-dir", str(pkg / "assets")]) == 0


def test_main_refuses_to_replace_a_directory_with_files(tmp_path, monkeypatch, capsys):
    root = snapshot(tmp_path / "snap", FILES[1:])
    monkeypatch.setattr(A, "fetch", lambda **kw: root)
    stale = tmp_path / "assets"
    (stale / "scenes").mkdir(parents=True)
    (stale / "scenes" / "other.xml").write_text("x")
    assert A.main(["--assets-dir", str(stale)]) == 1
    assert "holds files" in capsys.readouterr().err
    assert not stale.is_symlink()


def test_from_package_links_the_tree_another_package_uses(tmp_path, monkeypatch):
    pro = snapshot(tmp_path / "hf" / "snap", FILES[1:])
    site = tmp_path / "site"
    (site / "liberopro" / "liberopro").mkdir(parents=True)
    (site / "liberopro" / "__init__.py").write_text("")
    (site / "liberopro" / "liberopro" / "__init__.py").write_text("")
    (site / "liberopro" / "liberopro" / "assets").symlink_to(pro)
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setattr(
        A, "fetch", lambda **kw: pytest.fail("no download for libero-pro")
    )
    dest = tmp_path / "libero_assets"
    assert (
        A.main(["--assets-dir", str(dest), "--from-package", "liberopro.liberopro"])
        == 0
    )
    assert dest.resolve() == pro.resolve()


def test_a_missing_package_is_named(tmp_path, capsys):
    rc = A.main(
        ["--assets-dir", str(tmp_path / "a"), "--from-package", "no_such_pkg_xyz.x"]
    )
    assert rc == 1
    assert "no_such_pkg_xyz.x is not installed" in capsys.readouterr().err
