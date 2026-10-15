"""setup.sh libero's scene assets (robots/libero/assets.py): a partial snapshot is never
linked, a complete one is, an existing tree is kept, and libero-pro links LIBERO-PRO's tree."""

from __future__ import annotations

from pathlib import Path

import pytest

from pi_embodied_services.robots.libero import assets as A

FILES = [".gitattributes", A.PROBE, "textures/wood.png"]


def snapshot(root: Path, files: list[str]) -> Path:
    for f in files:
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text("x")
    return root


def hub(root: Path, listed: list[str] = FILES):
    calls = []

    def download(**kw):
        calls.append(kw)
        return str(root)

    return download, (lambda repo, rev: list(listed)), calls


def test_fetch_pins_the_revision_and_returns_a_complete_snapshot(tmp_path):
    root = snapshot(tmp_path / "snap", FILES[1:])
    download, files, calls = hub(root)
    assert A.fetch(download=download, list_files=files) == root
    assert calls == [
        {"repo_id": A.REPO, "repo_type": "dataset", "revision": A.REVISION}
    ]


def test_a_partial_snapshot_is_refused(tmp_path):
    root = snapshot(tmp_path / "snap", [A.PROBE])
    download, files, _ = hub(root)
    with pytest.raises(RuntimeError, match=r"1 of 3 files missing.*textures/wood\.png"):
        A.fetch(download=download, list_files=files)


def test_an_unlistable_hub_is_refused_not_trusted(tmp_path):
    """The offline fallback: the cached snapshot comes back, but nothing can verify it."""
    root = snapshot(tmp_path / "snap", FILES[1:])

    def offline(repo, rev):
        raise OSError("hub unreachable")

    with pytest.raises(RuntimeError, match="could not list"):
        A.fetch(download=lambda **kw: str(root), list_files=offline)


def test_main_links_the_package_dir_and_keeps_a_present_tree(tmp_path, monkeypatch):
    root = snapshot(tmp_path / "snap", FILES[1:])
    monkeypatch.setattr(A, "fetch", lambda: root)
    pkg = tmp_path / "libero" / "libero"
    pkg.mkdir(parents=True)
    (pkg / "assets").mkdir()  # an empty directory, as a wheel without assets leaves
    assert A.main(["--assets-dir", str(pkg / "assets")]) == 0
    assert (pkg / "assets").resolve() == root.resolve()
    assert A.present(pkg / "assets")
    # A second run keeps it without fetching.
    monkeypatch.setattr(A, "fetch", lambda: pytest.fail("fetched again"))
    assert A.main(["--assets-dir", str(pkg / "assets")]) == 0


def test_main_refuses_to_replace_a_directory_with_files(tmp_path, monkeypatch, capsys):
    root = snapshot(tmp_path / "snap", FILES[1:])
    monkeypatch.setattr(A, "fetch", lambda: root)
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
    monkeypatch.setattr(A, "fetch", lambda: pytest.fail("no download for libero-pro"))
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
