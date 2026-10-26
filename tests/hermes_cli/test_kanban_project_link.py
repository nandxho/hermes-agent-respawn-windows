"""Kanban <-> Projects integration: project-linked tasks get a deterministic
worktree path + branch instead of the random ``wt/<task-id>`` fallback."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli import projects_db as pdb


@pytest.fixture
def kanban_conn(tmp_path):
    c = kbc.connect(db_path=tmp_path / "kanban.db")
    try:
        yield c
    finally:
        c.close()


def _make_project(name="Web App", repo="/tmp/webapp"):
    with pdb.connect_closing() as pc:
        pid = pdb.create_project(pc, name=name, folders=[repo])
        return pdb.get_project(pc, pid)


def test_project_linked_task_gets_deterministic_worktree_and_branch(kanban_conn):
    proj = _make_project()
    tid = kb.create_task(kanban_conn, title="Add login", project_id=proj.slug)
    task = kb.get_task(kanban_conn, tid)

    assert task.project_id == proj.id
    assert task.workspace_kind == "worktree"
    # Worktree dir anchored under the project's primary repo, keyed on task id.
    assert task.workspace_path == os.path.join(proj.primary_path, ".worktrees", tid)
    # Deterministic branch: <slug>/<task-id>-<title-slug>. NOT a random wt/...
    assert task.branch_name == f"{proj.slug}/{tid}-add-login"
    assert not task.branch_name.startswith("wt/")


def test_explicit_branch_overrides_project_default(kanban_conn):
    proj = _make_project()
    tid = kb.create_task(
        kanban_conn,
        title="x",
        project_id=proj.slug,
        workspace_kind="worktree",
        branch_name="feature/custom",
    )
    task = kb.get_task(kanban_conn, tid)
    assert task.branch_name == "feature/custom"


def test_unlinked_task_unchanged(kanban_conn):
    tid = kb.create_task(kanban_conn, title="plain")
    task = kb.get_task(kanban_conn, tid)

    assert task.project_id is None
    assert task.workspace_kind == "scratch"
    # No branch is persisted — the worker still owns the wt/<id> fallback for
    # genuinely ad-hoc worktree tasks, but unlinked scratch tasks have none.
    assert task.branch_name is None


def _init_repo(repo: Path) -> Path:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "kanban@example.com"],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Kanban Test"],
        check=True, capture_output=True,
    )
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "init"], check=True, capture_output=True
    )
    return repo


def test_project_linked_task_resolves_outside_the_checkout(kanban_conn, tmp_path):
    """La ruta ``<repo>/.worktrees/<id>`` es marcador de nacimiento, no ubicación.

    25-oct-2026: las tarjetas ``project=`` materializaban el árbol dentro del
    checkout (una de las tres fuentes de los 5.7 GB anidados). El resolver lo
    redirige al target anclado al board y el repo del proyecto se recupera del
    git común del worktree, no de la forma del path.
    """
    repo = _init_repo(tmp_path / "repo")
    proj = _make_project(repo=str(repo))
    assert proj is not None
    tid = kb.create_task(kanban_conn, title="Add login", project_id=proj.slug)
    task = kb.get_task(kanban_conn, tid)
    assert task is not None

    # Marcador persistido al crear (lo consume _project_from_source_task).
    marker = Path(task.workspace_path)
    assert marker == Path(proj.primary_path) / ".worktrees" / tid

    ws = kbw.resolve_workspace(task)
    assert repo not in ws.parents
    assert ws.name == "wt" and ws.parent.name == tid
    assert (ws / "README.md").exists()
    assert not marker.exists()
    assert not (repo / ".worktrees").exists()

    # El dispatcher persiste el path resuelto (kanban_db_dispatch) …
    kbw.set_workspace_path(kanban_conn, tid, ws)
    stored = kb.get_task(kanban_conn, tid)
    assert stored is not None and Path(stored.workspace_path) == ws

    # … y aun así otro perfil puede recuperar proyecto + repo desde el board.
    recovered, recovered_repo = kb._project_from_source_task(kanban_conn, pdb, proj.id, tid)
    assert recovered is not None
    assert Path(recovered.primary_path).resolve() == repo.resolve()
    assert recovered_repo is not None
    assert Path(recovered_repo).resolve() == repo.resolve()


