"""Dispatcher knobs are re-read per tick: applying a brake must not need a restart.

``_KanbanDispatcher`` used to receive settings resolved once at boot. Lowering
``kanban.max_in_progress`` under load therefore did nothing until
``hermes gateway restart`` — and the workers are the gateway's children, so that
restart killed exactly the workers the new cap was meant to protect. Adjusting
the brake caused the wave the brake exists to prevent (t_51b36d2e).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway import kanban_watchers_dispatcher as kwd


def _settings(**overrides):
    base = dict(
        interval=60.0,
        max_spawn=None,
        max_in_progress=8,
        failure_limit=2,
        stale_timeout_seconds=0,
        reconcile_orphans=True,
        default_assignee=None,
        max_in_progress_per_profile=None,
    )
    base.update(overrides)
    return kwd._DispatcherSettings(**base)


def _kb_stub():
    return SimpleNamespace(DEFAULT_FAILURE_LIMIT=2)


def _dispatcher(monkeypatch, config: dict, *, initial=None):
    """Dispatcher whose per-tick provider resolves from ``config['kanban']``."""
    monkeypatch.setattr(kwd, "_board_slugs", lambda kb: ["default"])
    boot = initial or kwd._resolve_dispatcher_settings(dict(config["kanban"]), _kb_stub(), quiet=True)
    dispatcher = kwd._KanbanDispatcher(
        _kb_stub(),
        boot,
        settings_provider=lambda: kwd._resolve_dispatcher_settings(
            dict(config["kanban"]), _kb_stub(), quiet=True
        ),
    )
    seen: list[object] = []
    monkeypatch.setattr(
        dispatcher, "tick_once_for_board", lambda slug: seen.append(dispatcher.settings.max_in_progress)
    )
    return dispatcher, seen


def test_lowered_cap_applies_on_the_next_tick(monkeypatch):
    config = {"kanban": {"max_in_progress": 8}}
    dispatcher, seen = _dispatcher(monkeypatch, config)

    dispatcher.tick_once()
    assert seen == [8]

    # Operator edits config.yaml while the gateway runs (no restart).
    config["kanban"]["max_in_progress"] = 2
    dispatcher.tick_once()
    assert seen == [8, 2]
    assert dispatcher.settings.max_in_progress == 2


def test_reload_covers_the_other_brakes_too(monkeypatch):
    config = {"kanban": {"max_in_progress": 4, "failure_limit": 3,
                         "dispatch_stale_timeout_seconds": 900,
                         "dispatch_interval_seconds": 60}}
    dispatcher, _seen = _dispatcher(monkeypatch, config)

    config["kanban"].update(
        max_in_progress=1, failure_limit=1, dispatch_stale_timeout_seconds=300,
        dispatch_interval_seconds=15,
    )
    assert dispatcher.refresh_settings() is True
    assert dispatcher.settings.max_in_progress == 1
    assert dispatcher.settings.failure_limit == 1
    assert dispatcher.settings.stale_timeout_seconds == 300
    assert dispatcher.settings.interval == 15.0


def test_no_config_change_keeps_settings_and_reports_no_reload(monkeypatch):
    config = {"kanban": {"max_in_progress": 8}}
    dispatcher, _seen = _dispatcher(monkeypatch, config)
    same = dispatcher.settings

    assert dispatcher.refresh_settings() is False
    assert dispatcher.settings is same


def test_provider_failure_keeps_the_previous_values(monkeypatch):
    """A broken config must not silently widen a running dispatcher's cap."""
    dispatcher = kwd._KanbanDispatcher(
        _kb_stub(), _settings(max_in_progress=2),
        settings_provider=lambda: (_ for _ in ()).throw(RuntimeError("bad yaml")),
    )
    monkeypatch.setattr(kwd, "_board_slugs", lambda kb: ["default"])
    monkeypatch.setattr(dispatcher, "tick_once_for_board", lambda slug: None)

    dispatcher.tick_once()
    assert dispatcher.settings.max_in_progress == 2


def test_no_provider_keeps_boot_settings(monkeypatch):
    dispatcher = kwd._KanbanDispatcher(_kb_stub(), _settings(max_in_progress=3))
    assert dispatcher.refresh_settings() is False
    assert dispatcher.settings.max_in_progress == 3


def test_yaml_edit_on_disk_applies_while_the_dispatcher_runs(tmp_path, monkeypatch):
    """The production shape: the provider re-reads config.yaml, so an operator's
    edit lands on the next tick (this also exercises load_config's file-signature
    cache — a stale cache would pin the boot value forever)."""
    root = tmp_path / ".hermes"
    root.mkdir()
    cfg_path = root / "config.yaml"
    cfg_path.write_text("kanban:\n  max_in_progress: 8\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import config as config_mod

    config_mod._LOAD_CONFIG_CACHE.clear()

    def _provider():
        cfg = config_mod.load_config_readonly() or {}
        return kwd._resolve_dispatcher_settings(cfg.get("kanban", {}) or {}, _kb_stub(), quiet=True)

    dispatcher = kwd._KanbanDispatcher(_kb_stub(), _provider(), settings_provider=_provider)
    monkeypatch.setattr(kwd, "_board_slugs", lambda kb: ["default"])
    seen: list[object] = []
    monkeypatch.setattr(
        dispatcher, "tick_once_for_board", lambda slug: seen.append(dispatcher.settings.max_in_progress)
    )
    try:
        dispatcher.tick_once()
        cfg_path.write_text("kanban:\n  max_in_progress: 2\n", encoding="utf-8")
        dispatcher.tick_once()
    finally:
        config_mod._LOAD_CONFIG_CACHE.clear()

    assert seen == [8, 2]


def test_gateway_settings_inherit_the_root_cap_for_a_profile_without_the_key(tmp_path, monkeypatch):
    """Gateway-side resolver: profile ``kanban:`` without the key ⇒ root cap, not None."""
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "alice"
    profile.mkdir(parents=True)
    (root / "config.yaml").write_text("kanban:\n  max_in_progress: 5\n", encoding="utf-8")
    (profile / "config.yaml").write_text(
        "kanban:\n  dispatch_interval_seconds: 30\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(profile))
    from hermes_cli import config as config_mod

    config_mod._LOAD_CONFIG_CACHE.clear()
    try:
        settings = kwd._resolve_dispatcher_settings({"dispatch_interval_seconds": 30}, _kb_stub(), quiet=True)
        assert settings.max_in_progress == 5
        assert settings.interval == 30.0
    finally:
        config_mod._LOAD_CONFIG_CACHE.clear()


def test_gateway_settings_fall_back_to_derived_cap_when_nothing_is_capped(tmp_path, monkeypatch):
    """Unset everywhere ⇒ derived floor, never None (unbounded) — the macOS bug."""
    root = tmp_path / "other" / ".hermes"
    root.mkdir(parents=True)
    (root / "config.yaml").write_text("kanban:\n  dispatch_interval_seconds: 30\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    from hermes_cli import config as config_mod

    config_mod._LOAD_CONFIG_CACHE.clear()
    try:
        settings = kwd._resolve_dispatcher_settings({}, _kb_stub(), quiet=True)
        assert isinstance(settings.max_in_progress, int)
        assert settings.max_in_progress >= 1
    finally:
        config_mod._LOAD_CONFIG_CACHE.clear()
