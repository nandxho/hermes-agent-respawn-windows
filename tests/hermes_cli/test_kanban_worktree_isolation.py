"""Per-task worktree isolation for decompose siblings.

Decompose children used to inherit the root's literal ``workspace_path``,
so every sibling of a worktree-kind root pointed at the SAME checkout —
and ``_resolve_worktree_workspace``'s existing-checkout shortcut reused it
on whatever branch was there, letting sibling workers run concurrently in
one directory on one branch (cross-task provenance corruption, no lock).

Two-part fix under test:
- ``decompose_triage_task`` leaves worktree children's ``workspace_path``
  unset so each child materializes its own ``<board-root>/workspaces/<id>/wt``.
- ``_resolve_worktree_workspace`` falls back to a fresh per-task worktree
  when the requested path is occupied by another task's branch (heals
  pre-existing rows that still carry a shared path).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_graph import decompose_triage_task
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=Test User",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        check=True, capture_output=True, text=True,
    )


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True, capture_output=True, text=True,
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    return repo


def _add_worktree(repo: Path, target: Path, branch: str) -> Path:
    _git(repo, "worktree", "add", str(target), "-b", branch, "HEAD")
    return target


def test_decompose_worktree_children_get_own_workspace(kanban_home):
    with kbc.connect() as conn:
        root = kb.create_task(conn, title="build the feature", triage=True)
        conn.execute(
            "UPDATE tasks SET workspace_kind='worktree', "
            "workspace_path='/repo/.worktrees/root' WHERE id = ?",
            (root,),
        )
        conn.commit()

        child_ids = decompose_triage_task(
            conn,
            root,
            root_assignee="orchestrator",
            children=[
                {"title": "spec it", "assignee": "alice", "parents": []},
                {"title": "implement it", "assignee": "bob", "parents": [0]},
            ],
            author="decomposer",
        )
        assert child_ids is not None and len(child_ids) == 2

        for cid in child_ids:
            row = conn.execute(
                "SELECT workspace_kind, workspace_path FROM tasks WHERE id = ?",
                (cid,),
            ).fetchone()
            assert row["workspace_kind"] == "worktree"
            # Each child resolves its own <repo>/.worktrees/<child-id> at
            # dispatch; the root's literal path must never be shared.
            assert row["workspace_path"] is None




def test_resolve_worktree_falls_back_when_path_occupied(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    occupied = _add_worktree(repo, repo / ".worktrees" / "sibling", "wt/sibling")

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn,
            title="second sibling",
            workspace_kind="worktree",
            workspace_path=str(occupied),  # inherited shared/stale path
        )
        task = kb.get_task(conn, tid)

    workspace, branch = kbw._resolve_worktree_workspace(task)
    # The fallback tree must live OUTSIDE the checkout (25-oct-2026: nested
    # ``<repo>/.worktrees/*`` trees cost 5.7 GB and hid from `git status`).
    assert workspace.name == "wt"
    assert workspace.parent.name == tid
    assert repo not in workspace.parents
    assert (kanban_home / "kanban") in workspace.parents
    assert branch == f"wt/{tid}"
    # …and it must be materialized AT the returned path, not merely returned:
    # asserting the shape alone passes for a resolver that creates the tree
    # somewhere else.
    assert (workspace / "README.md").exists()
    # The sibling's checkout is untouched, still on its own branch.
    assert (occupied / "README.md").exists()
    head = subprocess.run(
        ["git", "-C", str(occupied), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert head == "wt/sibling"


def test_legacy_nested_path_is_never_materialized_inside_checkout(kanban_home, tmp_path):
    """Forma legada ``<repo>/.worktrees/<id>`` (166 filas del board himex).

    Hoy son tarjetas terminales, pero un reintento o reapertura vuelve a
    pasar por el resolver: sin esta redirección el árbol nace DENTRO de
    ``04_Desarrollo`` — el síntoma que esta tarjeta existe para cerrar.
    """
    repo = _make_repo(tmp_path)
    legacy = repo / ".worktrees" / "t_legacy01"

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn,
            title="tarjeta legada",
            workspace_kind="worktree",
            workspace_path=str(legacy),
        )
        task = kb.get_task(conn, tid)

    assert task is not None
    workspace, branch = kbw._resolve_worktree_workspace(task)

    assert repo not in workspace.parents
    assert (kanban_home / "kanban") in workspace.parents
    assert workspace.name == "wt" and workspace.parent.name == tid
    # Materializado en el path devuelto, fuera del checkout.
    assert (workspace / "README.md").exists()
    assert branch == f"wt/{tid}"
    # El path legado no existe: nada se creó dentro del repo.
    assert not legacy.exists()
    assert not (repo / ".worktrees").exists()


def test_existing_legacy_tree_is_moved_out_of_checkout(kanban_home, tmp_path):
    """Un árbol ya materializado en la forma legada se mueve, no se abandona.

    ``git worktree move`` conserva el trabajo sin commitear: mover no es
    borrar+recrear.
    """
    repo = _make_repo(tmp_path)

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn,
            title="tarjeta legada con árbol vivo",
            workspace_kind="worktree",
            workspace_path=str(repo),
        )
        legacy = repo / ".worktrees" / tid
        conn.execute(
            "UPDATE tasks SET workspace_path = ? WHERE id = ?", (str(legacy), tid)
        )
        conn.commit()
        task = kb.get_task(conn, tid)

    _add_worktree(repo, legacy, f"wt/{tid}")
    (legacy / "wip.txt").write_text("trabajo sin commitear\n", encoding="utf-8")

    assert task is not None
    workspace, branch = kbw._resolve_worktree_workspace(task)

    assert repo not in workspace.parents
    assert workspace.name == "wt" and workspace.parent.name == tid
    assert branch == f"wt/{tid}"
    # El árbol se movió con su contenido intacto.
    assert (workspace / "README.md").exists()
    assert (workspace / "wip.txt").read_text(encoding="utf-8") == "trabajo sin commitear\n"
    assert not legacy.exists()
    # git ya no lo registra dentro del checkout.
    listed = subprocess.run(
        ["git", "-C", str(repo), "worktree", "list", "--porcelain"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert str(legacy) not in listed
    assert str(workspace) in listed


def test_new_worktree_bases_on_trunk_not_checkout_head(kanban_home, tmp_path):
    """Una tarjeta nueva nace del tronco, no de lo que tenga el checkout.

    26-oct-2026: el checkout principal estaba en la rama de otra tarjeta
    (``feat/ui-portal-activacion-t_d0a95663``), así que el worktree de una
    tarjeta recién creada heredaba ese trabajo sin mergear y lo habría
    arrastrado a su PR.
    """
    repo = _make_repo(tmp_path)
    trunk_sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    _git(repo, "update-ref", "refs/remotes/origin/main", trunk_sha)

    # Otra tarjeta deja el checkout compartido en su rama, con un commit propio.
    _git(repo, "checkout", "-b", "feat/otra-tarjeta")
    (repo / "otra.txt").write_text("trabajo sin mergear\n", encoding="utf-8")
    _git(repo, "add", "otra.txt")
    _git(repo, "commit", "-m", "wip de otra tarjeta")

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn,
            title="tarjeta nueva",
            workspace_kind="worktree",
            workspace_path=str(repo),  # el dispatcher rellena el repo, no el árbol
        )
        task = kb.get_task(conn, tid)

    assert task is not None
    workspace, _ = kbw._resolve_worktree_workspace(task)

    def _head(path: Path) -> str:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    assert _head(workspace) == trunk_sha
    assert not (workspace / "otra.txt").exists()



