"""Tests for the self-updater (no network: fetch/download are injected)."""

import hashlib
import sys

import pytest

import updater as upd


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("v0.3.12", (0, 3, 12)),
    ("0.3.12", (0, 3, 12)),
    ("1", (1,)),
    ("0.3.0-dev", None),
    ("", None),
    ("latest", None),
])
def test_parse_version(text, expected):
    assert upd.parse_version(text) == expected


@pytest.mark.parametrize("remote,local,newer", [
    ("v0.3.12", "0.3.11", True),
    ("v0.3.10", "0.3.9", True),     # numeric, not string, comparison
    ("v0.3.9", "0.3.10", False),
    ("v0.3.5", "0.3.5", False),
    ("v0.4", "0.3.99", True),
    ("v0.3.0", "0.3", False),       # padding
    ("v0.3.1", "0.3.0-dev", False),  # dev builds never update
    ("garbage", "0.3.1", False),
])
def test_is_newer(remote, local, newer):
    assert upd.is_newer(remote, local) is newer


def test_updates_supported_from_source(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert upd.updates_supported("0.3.5")[0] is False


def test_updates_supported_dev_build(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    ok, reason = upd.updates_supported("0.3.0-dev")
    assert not ok and "local build" in reason
    assert upd.updates_supported("0.3.5") == (True, "")


# ---------------------------------------------------------------------------
# Release parsing / verification
# ---------------------------------------------------------------------------
def make_release(tag="v0.3.9", payload=b"new exe", name="Sauron.exe", digest=True):
    asset = {"name": name, "browser_download_url": "https://example.invalid/x",
             "size": len(payload)}
    if digest:
        asset["digest"] = "sha256:" + hashlib.sha256(payload).hexdigest()
    return {"tag_name": tag, "assets": [{"name": "other.zip"}, asset]}


def test_pick_asset():
    rel = make_release()
    assert upd.pick_asset(rel)["name"] == "Sauron.exe"
    assert upd.pick_asset(make_release(name="Other.exe")) is None
    assert upd.pick_asset({}) is None


def test_verify_download():
    payload = b"abc"
    asset = make_release(payload=payload)["assets"][1]
    upd.verify_download(asset, 3, hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError, match="size"):
        upd.verify_download(asset, 4, hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError, match="SHA-256"):
        upd.verify_download(asset, 3, "00" * 32)
    no_digest = make_release(payload=payload, digest=False)["assets"][1]
    upd.verify_download(no_digest, 3, "anything")


# ---------------------------------------------------------------------------
# Updater flow with fake exe files
# ---------------------------------------------------------------------------
@pytest.fixture
def exe(tmp_path):
    path = tmp_path / "Sauron.exe"
    path.write_bytes(b"old exe")
    return path


def fake_download(payload):
    def download(asset, dest):
        with open(dest, "wb") as f:
            f.write(payload)
    return download


def test_check_once_stages_newer_release(exe):
    u = upd.Updater("0.3.8", str(exe), 6, fetch=lambda: make_release("v0.3.9"),
                    download=fake_download(b"new exe"))
    assert u.check_once() is True
    assert u.ready and u.new_version == "0.3.9"
    assert (exe.parent / "Sauron.exe.new").read_bytes() == b"new exe"


def test_check_once_ignores_same_or_older(exe):
    calls = []
    u = upd.Updater("0.3.9", str(exe), 6, fetch=lambda: make_release("v0.3.9"),
                    download=lambda a, d: calls.append(d))
    assert u.check_once() is False
    assert not u.ready and calls == []


def test_check_once_without_asset(exe):
    u = upd.Updater("0.3.8", str(exe), 6,
                    fetch=lambda: make_release("v0.3.9", name="nope.exe"),
                    download=fake_download(b"x"))
    assert u.check_once() is False and not u.ready


def test_apply_swaps_and_keeps_old(exe):
    u = upd.Updater("0.3.8", str(exe), 6, fetch=lambda: make_release("v0.3.9"),
                    download=fake_download(b"new exe"))
    assert u.apply() is False  # nothing staged yet
    u.check_once()
    assert u.apply() is True
    assert exe.read_bytes() == b"new exe"
    assert (exe.parent / "Sauron.exe.old").read_bytes() == b"old exe"
    assert not (exe.parent / "Sauron.exe.new").exists()
    upd.cleanup_old(str(exe))
    assert not (exe.parent / "Sauron.exe.old").exists()


def test_swap_replaces_stale_old_file(exe):
    (exe.parent / "Sauron.exe.old").write_bytes(b"ancient")
    (exe.parent / "Sauron.exe.new").write_bytes(b"new")
    assert upd.swap_in(str(exe)) is True
    assert exe.read_bytes() == b"new"
    assert (exe.parent / "Sauron.exe.old").read_bytes() == b"old exe"


def test_swap_without_staged_file(exe):
    assert upd.swap_in(str(exe)) is False
    assert exe.read_bytes() == b"old exe"


def test_failed_apply_disables_further_attempts(exe, monkeypatch):
    u = upd.Updater("0.3.8", str(exe), 6, fetch=lambda: make_release("v0.3.9"),
                    download=fake_download(b"new exe"))
    u.check_once()
    monkeypatch.setattr(upd, "swap_in", lambda path: False)
    assert u.apply() is False
    assert not u.ready
    assert exe.read_bytes() == b"old exe"
