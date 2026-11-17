"""``kanban.max_in_progress`` inheritance: profile section without the key ⇒ root value.

Profile configs are loaded on their own — there is no deep merge with the root
home's ``config.yaml`` — so a profile that carries a ``kanban:`` block for
anything else resolved ``max_in_progress=None`` while the operator had set a
global cap. ``reviewer``, ``backend-eng`` and ``frontend-eng`` all ran unbounded
on one swap-thrashing host that way (t_51b36d2e).

Resolution order (see ``configured_max_in_progress``): explicit profile value →
root home's value → memory-derived default (never None).
"""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db_dispatch as kbd


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _clear_config_cache():
    from hermes_cli import config as config_mod

    config_mod._LOAD_CONFIG_CACHE.clear()


@pytest.fixture
def profile_tree(tmp_path, monkeypatch):
    """Root home with a global cap, plus a named profile with a ``kanban:`` block."""
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "alice"
    _write(root / "config.yaml", "kanban:\n  max_in_progress: 5\n")
    _write(profile / "config.yaml", "kanban:\n  dispatch_interval_seconds: 30\n")
    monkeypatch.setenv("HERMES_HOME", str(profile))
    _clear_config_cache()
    yield root, profile
    _clear_config_cache()


def test_profile_section_without_the_key_inherits_the_root_cap(profile_tree):
    _root, _profile = profile_tree
    assert kbd.inherited_max_in_progress() == 5
    assert kbd.configured_max_in_progress() == 5
    assert kbd.resolve_max_in_progress(kbd.configured_max_in_progress()) == 5


def test_profile_explicit_cap_wins_over_the_root(profile_tree):
    _root, profile = profile_tree
    _write(profile / "config.yaml", "kanban:\n  max_in_progress: 3\n")
    _clear_config_cache()
    assert kbd.inherited_max_in_progress() == 5
    assert kbd.configured_max_in_progress() == 3


def test_no_kanban_section_at_all_still_inherits(profile_tree):
    """A config with no ``kanban:`` block inherits too (defaults leave the key None)."""
    _root, profile = profile_tree
    _write(profile / "config.yaml", "providers: {}\n")
    _clear_config_cache()
    assert kbd.configured_max_in_progress() == 5


def test_root_home_does_not_inherit_from_itself(tmp_path, monkeypatch):
    root = tmp_path / "other" / ".hermes"
    _write(root / "config.yaml", "kanban:\n  dispatch_interval_seconds: 30\n")
    monkeypatch.setenv("HERMES_HOME", str(root))
    _clear_config_cache()
    try:
        assert kbd.inherited_max_in_progress() is None
        # Nothing to inherit → the memory-derived floor, still never None.
        assert kbd.resolve_max_in_progress(kbd.configured_max_in_progress()) >= 1
    finally:
        _clear_config_cache()


def test_invalid_root_value_is_not_inherited(profile_tree):
    """``0``/``-1`` in the root config is not a cap: fall through to the derived default."""
    root, _profile = profile_tree
    _write(root / "config.yaml", "kanban:\n  max_in_progress: 0\n")
    _clear_config_cache()
    assert kbd.inherited_max_in_progress() is None
    assert kbd.configured_max_in_progress() is None
