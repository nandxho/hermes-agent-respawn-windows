"""Exits from the two board columns that had no CLI way out (07-nov-2026).

Measured on two live cards:

* ``t_2235fc5f`` (top priority) entered ``scheduled`` on 02-nov, ran once and
  never moved again: ``promote`` answered "only applies to 'todo' or
  'blocked'", ``reclaim`` answered "not running" (it was ``scheduled``, not
  ``running``) and ``edit`` has no ``--status``. Archiving and re-creating by
  hand was the only way out — five days with the board's top card invisible.
* ``t_b7d6b063`` sat in ``triage`` with all five parents ``done`` and its goal
  already met; ``complete`` answered "unknown id or terminal state". Only
  ``archive`` worked, which erases the difference between "fulfilled" and
  "discarded".

These tests pin both exits and the fingerprint cleanup that makes a freed
``scheduled`` card readable again (a dead worker's ``worker_started_at`` used to
survive every exit path).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pytest

from hermes_cli import kanban as kb_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


def _col(conn, task_id: str, column: str):
    """Read a raw column: the Task dataclass doesn't expose every one of them."""
    return conn.execute(
        f"SELECT {column} FROM tasks WHERE id=?", (task_id,),
    ).fetchone()[0]


def _parked_with_dead_worker(conn, *, hours_ago: float = 30.0) -> str:
    """``scheduled`` carrying the fingerprints of the run that stranded it.

    Mirrors t_2235fc5f: the card was claimed once (open run), then parked while
    ``worker_pid``/``worker_started_at`` stayed behind on the row.
    """
    tid = kb.create_task(conn, title="deuda de 222 órdenes", assignee="backend-eng")
    claim = kb.claim_task(conn, tid)
    assert claim is not None, "fixture needs a claimed card"
    conn.execute(
        "UPDATE tasks SET status='scheduled', worker_started_at=?, claim_lock=NULL, "
        "claim_expires=NULL WHERE id=?",
        (int(time.time() - hours_ago * 3600), tid),
    )
    conn.commit()
    assert kb.get_task(conn, tid).status == "scheduled"
    return tid


# ---------------------------------------------------------------------------
# promote: scheduled
# ---------------------------------------------------------------------------


def test_promote_frees_scheduled_and_wipes_the_dead_worker_fingerprint(conn):
    tid = _parked_with_dead_worker(conn)
    run_id = kb.get_task(conn, tid).current_run_id

    ok, err = kb.promote_task(
        conn, tid, actor="tester", reason="worker muerto, la devuelvo a ready",
    )

    assert ok, err
    task = kb.get_task(conn, tid)
    assert task.status == "ready"
    assert task.worker_pid is None
    assert _col(conn, tid, "worker_started_at") is None
    assert task.current_run_id is None
    # The leaked run is closed, so it can't keep reading as "a worker is on it".
    run = conn.execute(
        "SELECT status, ended_at FROM task_runs WHERE id=?", (run_id,),
    ).fetchone()
    assert run["ended_at"] is not None
    assert run["status"] in ("reclaimed", "scheduled")
    # Audited, with the source column named.
    ev = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='promoted_manual' "
        "ORDER BY id DESC LIMIT 1", (tid,),
    ).fetchone()
    assert ev is not None
    assert "scheduled" in ev["payload"]
    assert "worker muerto" in ev["payload"]


def test_promote_scheduled_requires_an_audited_reason(conn):
    tid = _parked_with_dead_worker(conn)
    ok, err = kb.promote_task(conn, tid, actor="tester")
    assert not ok
    assert err and "reason" in err
    assert kb.get_task(conn, tid).status == "scheduled"
    # dry-run must predict the same refusal, not a phantom success.
    ok, err = kb.promote_task(conn, tid, actor="tester", dry_run=True)
    assert not ok and err and "reason" in err


def test_promote_scheduled_dry_run_does_not_mutate(conn):
    tid = _parked_with_dead_worker(conn)
    ok, err = kb.promote_task(conn, tid, actor="tester", reason="revisión", dry_run=True)
    assert ok and err is None
    assert kb.get_task(conn, tid).status == "scheduled"


def test_promote_scheduled_still_refuses_an_unfinished_parent(conn):
    parent = kb.create_task(conn, title="padre abierto", assignee="setup")
    child = _parked_with_dead_worker(conn)
    kb.link_tasks(conn, parent, child)
    ok, err = kb.promote_task(conn, child, actor="tester", reason="recovery")
    assert not ok and err and parent in err
    assert kb.get_task(conn, child).status == "scheduled"


# ---------------------------------------------------------------------------
# promote: triage
# ---------------------------------------------------------------------------


def test_promote_frees_triage_with_reason(conn):
    tid = kb.create_task(conn, title="decisión pendiente", assignee="planner")
    conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (tid,))
    conn.commit()

    ok, err = kb.promote_task(conn, tid, actor="tester", reason="el spec ya está claro")
    assert ok, err
    assert kb.get_task(conn, tid).status == "ready"


def test_promote_triage_requires_reason(conn):
    tid = kb.create_task(conn, title="decisión", assignee="planner")
    conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (tid,))
    conn.commit()
    ok, err = kb.promote_task(conn, tid, actor="tester")
    assert not ok and err and "reason" in err
    assert kb.get_task(conn, tid).status == "triage"


def test_promote_names_every_accepted_column_when_refusing(conn):
    tid = kb.create_task(conn, title="ya cerrada", assignee="setup")
    conn.execute("UPDATE tasks SET status='done' WHERE id=?", (tid,))
    conn.commit()
    ok, err = kb.promote_task(conn, tid, actor="tester", reason="x")
    assert not ok
    assert err and "scheduled" in err and "triage" in err


# ---------------------------------------------------------------------------
# complete: triage
# ---------------------------------------------------------------------------


def test_complete_triage_closes_as_done_with_its_summary(conn):
    tid = kb.create_task(conn, title="decidir si se parte en dos", assignee="planner")
    conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (tid,))
    conn.commit()

    assert kb.complete_task(
        conn, tid, summary="Resuelto: no se parte, entra como una sola tarjeta.",
    )

    task = kb.get_task(conn, tid)
    assert task.status == "done"
    assert task.completed_at is not None
    run = conn.execute(
        "SELECT outcome, summary FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()
    assert run["outcome"] == "completed"
    assert "una sola tarjeta" in (run["summary"] or "")


def test_complete_triage_still_demands_evidence(conn):
    tid = kb.create_task(conn, title="decisión sin evidencia", assignee="planner")
    conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (tid,))
    conn.commit()
    with pytest.raises(kb.EmptyCompletionError):
        kb.complete_task(conn, tid)
    assert kb.get_task(conn, tid).status == "triage"


def test_completing_a_triage_parent_promotes_its_children(conn):
    parent = kb.create_task(conn, title="decisión", assignee="planner")
    child = kb.create_task(conn, title="hija", parents=[parent], assignee="backend-eng")
    conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (parent,))
    conn.commit()
    assert kb.get_task(conn, child).status == "todo"

    assert kb.complete_task(conn, parent, summary="Decidido: la hija arranca.")
    assert kb.get_task(conn, child).status == "ready"


# ---------------------------------------------------------------------------
# unblock: no dead-worker residue either
# ---------------------------------------------------------------------------


def test_unblock_of_scheduled_wipes_the_dead_worker_columns(conn):
    tid = _parked_with_dead_worker(conn)
    assert kb.unblock_task(conn, tid)
    task = kb.get_task(conn, tid)
    assert task.status == "ready"
    assert task.worker_pid is None
    assert _col(conn, tid, "worker_started_at") is None


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def _promote_ns(task_id, *, ids=None, reason=None, dry_run=False, as_json=False):
    return argparse.Namespace(
        task_id=task_id, reason=list(reason or []), ids=list(ids or []) or None,
        force=False, dry_run=dry_run, json=as_json,
    )


def _complete_ns(task_id, *, ids=None, result=None, summary=None, metadata=None):
    return argparse.Namespace(
        task_ids=[task_id, *list(ids or [])], result=result,
        summary=summary, metadata=metadata, force=False, json=False,
    )


def test_cli_promote_frees_a_scheduled_card(kanban_home, capsys):
    with kbc.connect() as conn:
        tid = _parked_with_dead_worker(conn)
    rc = kb_cli._cmd_promote(
        _promote_ns(tid, reason=["worker", "muerto,", "la", "devuelvo", "a", "ready"]),
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "scheduled" not in out  # "Promoted t_x -> ready: worker muerto..."
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "ready"


def test_cli_promote_without_reason_explains_the_requirement(kanban_home, capsys):
    with kbc.connect() as conn:
        tid = _parked_with_dead_worker(conn)
    rc = kb_cli._cmd_promote(_promote_ns(tid))
    err = capsys.readouterr().err
    assert rc == 1
    assert "reason" in err
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "scheduled"


def test_cli_complete_of_triage_succeeds_and_names_status_when_it_cannot(
    kanban_home, capsys,
):
    with kbc.connect() as conn:
        triage_id = kb.create_task(conn, title="decisión cumplida", assignee="planner")
        conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (triage_id,))
        parked_id = _parked_with_dead_worker(conn)
    rc = kb_cli._cmd_complete(
        _complete_ns(triage_id, summary="Cumplida: la decisión ya se tomó."),
    )
    assert rc == 0, capsys.readouterr().err
    with kbc.connect() as conn:
        assert kb.get_task(conn, triage_id).status == "done"

    rc = kb_cli._cmd_complete(_complete_ns(parked_id, summary="no aplica"))
    err = capsys.readouterr().err
    assert rc == 1
    # The old text said "unknown id or terminal state" and sent operators
    # hunting a bug instead of the promote verb.
    assert "'scheduled'" in err and "promote" in err
    with kbc.connect() as conn:
        assert kb.get_task(conn, parked_id).status == "scheduled"
