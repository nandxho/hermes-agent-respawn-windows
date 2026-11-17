"""Memory-aware kanban dispatch guard (OOF-30 / OOF-77).

Two production incidents shared the same failure shape: no
``kanban.max_in_progress`` configured, a busy board, and a small hosted VM —
the dispatcher fanned out 26-31 concurrent workers, the host swap-thrashed,
and the whole machine (dashboard included) became unreachable.

Covers the two safeguards added in response:

1. :func:`hermes_cli.kanban_db_dispatch.derive_default_max_in_progress` /
   :func:`hermes_cli.kanban_db_dispatch.resolve_max_in_progress` — memory-derived
   default global concurrency cap when the operator never set one. Fails CLOSED:
   an unreadable memory sample yields the conservative floor, never "uncapped".
2. The live memory-pressure guard inside ``dispatch_once`` — critical
   pressure spawns nothing; elevated pressure spawns at most one; unknown
   imposes no pressure restriction (the cap from #1 still bounds it).
3. Cross-platform sampling — ``sample_memory()`` must report a real level on
   macOS too (procfs is Linux-only), otherwise #2 never fires on the host that
   actually swap-thrashed (t_51b36d2e).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from gateway import lifecycle_ledger
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


GIB = 1024 * 1024  # KiB per GiB


# ---------------------------------------------------------------------------
# derive_default_max_in_progress / resolve_max_in_progress
# ---------------------------------------------------------------------------


def test_derived_cap_small_vm_floors_at_two():
    # 1 GiB VM (the incident shape): 1024 // 512 = 2 workers.
    assert kbd.derive_default_max_in_progress({"mem_total_kib": 1 * GIB}) == 2
    # Even tiny VMs keep a floor of 2 so boards still make progress.
    assert kbd.derive_default_max_in_progress({"mem_total_kib": GIB // 4}) == 2




def test_derived_cap_fails_closed_without_memtotal():
    """An unmeasurable host must not mean 'no cap' — it means the floor.

    macOS used to take this branch on every tick (``sample_memory()`` read
    procfs only), so ``resolve_max_in_progress(None)`` returned None and three
    dispatchers ran unbounded on a host already in swap-thrash (t_51b36d2e).
    """
    assert kbd.derive_default_max_in_progress({}) == kbd.DERIVED_MAX_IN_PROGRESS_FLOOR
    assert kbd.derive_default_max_in_progress({"mem_total_kib": 0}) == kbd.DERIVED_MAX_IN_PROGRESS_FLOOR
    assert kbd.derive_default_max_in_progress({"mem_total_kib": -5}) == kbd.DERIVED_MAX_IN_PROGRESS_FLOOR
    assert kbd.derive_default_max_in_progress({"mem_total_kib": True}) == kbd.DERIVED_MAX_IN_PROGRESS_FLOOR
    assert kbd.derive_default_max_in_progress(
        {"mem_total_kib": "1048576"}
    ) == kbd.DERIVED_MAX_IN_PROGRESS_FLOOR


def test_resolve_max_in_progress_explicit_config_wins(monkeypatch):
    monkeypatch.setattr(
        kbd, "_system_memory_sample", lambda: {"mem_total_kib": 1 * GIB}
    )
    # Operator said 6 on a 1 GiB box — their call, even above the derived 2.
    assert kbd.resolve_max_in_progress(6) == 6
    assert kbd.resolve_max_in_progress(1) == 1


def test_resolve_max_in_progress_derives_when_unset(monkeypatch):
    monkeypatch.setattr(
        kbd, "_system_memory_sample", lambda: {"mem_total_kib": 1 * GIB}
    )
    assert kbd.resolve_max_in_progress(None) == 2


def test_resolve_max_in_progress_unset_and_unknown_memory_uses_floor(monkeypatch):
    """Unset config + unreadable memory resolves to the floor, never to None."""
    monkeypatch.setattr(kbd, "_system_memory_sample", dict)
    assert kbd.resolve_max_in_progress(None) == kbd.DERIVED_MAX_IN_PROGRESS_FLOOR


# ---------------------------------------------------------------------------
# _memory_pressure_level
# ---------------------------------------------------------------------------


def test_pressure_level_ok_elevated_critical_from_a_real_sample():
    """A non-empty sample always yields a real level — never 'unknown'."""
    total = 1 * GIB
    assert kbd._memory_pressure_level(
        {"mem_available_kib": total // 2, "mem_total_kib": total}
    ) == "ok"
    assert kbd._memory_pressure_level(
        {"mem_available_kib": 100 * 1024, "mem_total_kib": total}
    ) == "elevated"
    assert kbd._memory_pressure_level(
        {"mem_available_kib": 32 * 1024, "mem_total_kib": total}
    ) == "critical"


def test_pressure_level_unknown_only_without_any_sample():
    assert kbd._memory_pressure_level({}) == "unknown"


def test_sample_memory_is_cross_platform(monkeypatch):
    """``sample_memory()`` reports real numbers where procfs is absent (macOS)."""
    monkeypatch.setattr(lifecycle_ledger, "_proc_fields", lambda *a, **k: {})
    monkeypatch.setattr(
        lifecycle_ledger, "_host_memory_sample",
        lambda: {"mem_total_kib": 36 * GIB, "mem_available_kib": 12 * GIB},
    )
    monkeypatch.setattr(lifecycle_ledger, "_self_rss_kib", lambda: 4096)
    sample = lifecycle_ledger.sample_memory()

    assert sample["mem_total_kib"] == 36 * GIB
    assert sample["mem_available_kib"] == 12 * GIB
    assert sample["rss_kib"] == 4096
    assert kbd._memory_pressure_level(sample) == "ok"


def test_sample_memory_still_prefers_procfs(monkeypatch):
    """Linux keeps the fork-free procfs path (no probe subprocesses)."""
    procfs = {
        "/proc/self/status": {"rss_kib": 2048},
        "/proc/meminfo": {
            "mem_total_kib": 4 * GIB,
            "mem_available_kib": 3 * GIB,
            "SwapTotal": 1024,
            "SwapFree": 256,
        },
    }
    monkeypatch.setattr(lifecycle_ledger, "_proc_fields", lambda path, wanted: dict(procfs[path]))
    monkeypatch.setattr(
        lifecycle_ledger, "_host_memory_sample",
        lambda: pytest.fail("procfs sample should short-circuit the fallback"),
    )
    sample = lifecycle_ledger.sample_memory()

    assert sample == {
        "rss_kib": 2048,
        "mem_total_kib": 4 * GIB,
        "mem_available_kib": 3 * GIB,
        "swap_used_kib": 768,
    }


def test_vm_stat_and_swapusage_parsing():
    """Darwin text parsing: pages × page size, swap units normalized to KiB."""
    vm_stat = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free:                               100.\n"
        "Pages active:                            2000.\n"
        "Pages inactive:                          3000.\n"
        "Pages speculative:                        500.\n"
        "Pages wired down:                         800.\n"
    )
    counters = lifecycle_ledger._parse_vm_stat(vm_stat)
    assert counters["page_size"] == 16384
    assert counters["pages free"] == 100
    assert counters["pages inactive"] == 3000
    assert counters["pages speculative"] == 500

    assert lifecycle_ledger._parse_swapusage(
        "total = 27648.00M  used = 26858.12M  free = 789.88M  (encrypted)"
    ) == {
        "swap_total_kib": 27648 * 1024,
        # Fractional MiB survives the conversion (truncated, like the rest of the sampler).
        "swap_used_kib": int(26858.12 * 1024),
    }
    assert lifecycle_ledger._parse_swapusage("garbage") == {}






# ---------------------------------------------------------------------------
# dispatch_once under pressure
# ---------------------------------------------------------------------------


def _pressure_sample(level: str) -> dict:
    total = 1 * GIB
    if level == "critical":
        return {"mem_available_kib": 32 * 1024, "mem_total_kib": total}
    if level == "elevated":
        return {"mem_available_kib": 100 * 1024, "mem_total_kib": total}
    return {"mem_available_kib": total // 2, "mem_total_kib": total}


def test_dispatch_spawns_nothing_under_critical_pressure(
    kanban_home, all_assignees_spawnable, monkeypatch, caplog,
):
    monkeypatch.setattr(
        kbd, "_system_memory_sample", lambda: _pressure_sample("critical")
    )
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with caplog.at_level(logging.WARNING), kbc.connect() as conn:
        for title in ("a", "b", "c"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)

    assert not spawns
    assert not res.spawned
    assert res.memory_pressure == "critical"
    # The operator has to be able to see WHY nothing spawned.
    assert any(
        rec.levelno == logging.WARNING and "memory pressure is critical" in rec.getMessage()
        for rec in caplog.records
    ), [rec.getMessage() for rec in caplog.records]


def test_dispatch_critical_pressure_defers_not_drops(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """Tasks skipped under pressure stay 'ready' and spawn once memory clears."""
    sample = {"value": _pressure_sample("critical")}
    monkeypatch.setattr(kbd, "_system_memory_sample", lambda: sample["value"])
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kbc.connect() as conn:
        task = kb.create_task(conn, title="a", assignee="alice")
        kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        assert not spawns
        row = kb.get_task(conn, task)
        assert row is not None and row.status == "ready"

        sample["value"] = _pressure_sample("ok")
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)

    assert spawns == [task]
    assert res.memory_pressure is None


def test_dispatch_elevated_pressure_spawns_at_most_one(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    monkeypatch.setattr(
        kbd, "_system_memory_sample", lambda: _pressure_sample("elevated")
    )
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kbc.connect() as conn:
        for title in ("a", "b", "c"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)

    assert len(spawns) == 1
    assert res.memory_pressure == "elevated"


def test_dispatch_elevated_pressure_does_not_widen_tighter_budget(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """A caller cap already at 0 remaining must not be widened to 1."""
    monkeypatch.setattr(
        kbd, "_system_memory_sample", lambda: _pressure_sample("elevated")
    )
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kbc.connect() as conn:
        running = kb.create_task(conn, title="running", assignee="alice")
        kb.claim_task(conn, running)
        kb.create_task(conn, title="ready", assignee="bob")
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn, max_in_progress=1)

    assert not spawns
    assert not res.spawned


def test_dispatch_unknown_pressure_imposes_no_restriction(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    monkeypatch.setattr(kbd, "_system_memory_sample", dict)
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kbc.connect() as conn:
        for title in ("a", "b", "c"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)

    assert len(spawns) == 3
    assert res.memory_pressure is None


def test_dispatch_critical_pressure_still_runs_reclaim_bookkeeping(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """The guard must only stop NEW spawns — reclaim/promotion still run."""
    monkeypatch.setattr(
        kbd, "_system_memory_sample", lambda: _pressure_sample("critical")
    )
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="alice")
        child = kb.create_task(
            conn, title="child", assignee="alice", parents=[parent],
        )
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (parent,))
        res = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 42)
        row = kb.get_task(conn, child)

    # Promotion (todo -> ready once parents are done) happened despite the
    # spawn freeze.
    assert res.memory_pressure == "critical"
    assert row is not None
    assert row.status == "ready"
