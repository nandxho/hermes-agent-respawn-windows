"""The publication gate, exercised against real git repos and a shimmed ``gh``.

The incident this closes: five cards were marked ``done`` while their commits lived
only in the local clone's ``.git``. Every test below therefore drives real git — a
bare ``origin``, a clone, a worktree on the card's branch — instead of mocking the
predicate, so the assertions are about behaviour, not about the implementation.
"""
import json
import os
import subprocess
import sys

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect


def _git(*args, cwd):
    proc = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True,
                          check=False)
    assert proc.returncode == 0, f"git {' '.join(args)} failed: {proc.stderr}"
    return proc.stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A bare ``origin`` plus a clone holding a ``wt/t_*`` worktree branch.

    ``origin`` is a local path so ``ls-remote`` answers offline and the test stays
    hermetic; the repo slug for the PR check comes from the board config override.
    """
    origin = tmp_path / "origin.git"
    origin.mkdir()
    _git("init", "--bare", "-b", "main", ".", cwd=origin)

    work = tmp_path / "work"
    work.mkdir()
    _git("init", "-b", "main", ".", cwd=work)
    _git("config", "user.email", "t@example.com", cwd=work)
    _git("config", "user.name", "Test", cwd=work)
    (work / "README.md").write_text("base\n")
    _git("add", "-A", cwd=work)
    _git("commit", "-m", "base", cwd=work)
    _git("remote", "add", "origin", str(origin), cwd=work)
    _git("push", "-u", "origin", "main", cwd=work)
    _git("fetch", "origin", cwd=work)

    # ``gh`` shim: prints the merged-PR list handed to it via $FAKE_PRS.
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    gh.write_text(f"#!{sys.executable}\nimport os\nprint(os.environ.get('FAKE_PRS', '[]'))\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("FAKE_PRS", "[]")

    # Sandbox the board. A dispatched worker carries HERMES_KANBAN_DB /
    # HERMES_KANBAN_BOARD / HERMES_KANBAN_WORKSPACES_ROOT pointing at the LIVE board,
    # and those win over HERMES_HOME when a board path is resolved — a test run
    # inside a worker would otherwise write its cards onto the real board.
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_WORKSPACES_ROOT",
                "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "kanban-home"))
    monkeypatch.setenv("HERMES_KANBAN_PUBLICATION_GATE", "1")
    monkeypatch.setenv("HERMES_KANBAN_PUBLICATION_REPO", "acme/repo")
    monkeypatch.delenv("HERMES_KANBAN_PUBLICATION_REMOTE", raising=False)

    board_db = kb.kanban_db_path()
    assert str(tmp_path) in str(board_db), f"refusing to point the tests at {board_db}"
    kb.init_db()
    return {"origin": origin, "work": work, "root": tmp_path, "board_db": board_db}


def _worktree_with_commit(repo, label, branch=None, message="work"):
    """A worktree on the card's branch holding one commit that is not on origin."""
    branch = branch or f"wt/{label}"
    wt = repo["root"] / f"wt-{label}"
    _git("worktree", "add", "-b", branch, str(wt), cwd=repo["work"])
    (wt / f"{label}.txt").write_text("done\n")
    _git("add", "-A", cwd=wt)
    _git("commit", "-m", message, cwd=wt)
    return wt, branch


def _card(repo, label, *, worktree=None, branch=None, kind="worktree",
          contract=None, assignee=None):
    """Create the card and return the id the board assigned it.

    ``create_task`` mints its own id, so the returned value is the only handle that
    refers to the row: completing any other id is a silent no-op that would make
    these tests pass for the wrong reason.
    """
    with connect() as conn:
        return kb.create_task(
            conn, title=f"card {label}", body="work",
            assignee=assignee, workspace_kind=kind,
            workspace_path=str(worktree) if worktree else None,
            branch_name=branch, completion_contract=contract,
        )


def _complete(task_id, **kw):
    with connect() as conn:
        return kb.complete_task(conn, task_id, result="did the work",
                                summary="did the work", **kw)


def _status(task_id):
    with connect() as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return row["status"] if row else None


def _events(task_id, kind=None):
    with connect() as conn:
        sql = "SELECT kind, payload FROM task_events WHERE task_id = ?"
        args = [task_id]
        if kind:
            sql += " AND kind = ?"
            args.append(kind)
        return [dict(zip(("kind", "payload"), row)) for row in conn.execute(sql, args)]


def test_unpublished_branch_refuses_the_close(repo):
    """The incident itself: work committed locally, never pushed, never merged."""
    wt, branch = _worktree_with_commit(repo, "t_aaa")
    card = _card(repo, "t_aaa", worktree=wt, branch=branch)

    with pytest.raises(kb.PublicationGateError) as err:
        _complete(card)

    receipt = err.value.receipt
    assert receipt["branch"] == branch
    assert [m["check"] for m in receipt["missing"]] == ["branch_published", "merged_pr"]
    assert f"push -u origin {branch}" in receipt["missing"][0]["fix"]
    # Actionable, and the exemption route is named rather than left to guesswork.
    assert "exempt" in receipt["remedy"]
    # Refused before the write txn: the card is untouched and the refusal is on the log.
    assert _status(card) == "ready"
    blocked = _events(card, "publication_gate_blocked")
    assert len(blocked) == 1
    assert json.loads(blocked[0]["payload"])["ok"] is False


def test_pushed_branch_without_a_merged_pr_still_refuses(repo):
    """Pushing is not landing: the card promises a merged PR, so an open one blocks."""
    wt, branch = _worktree_with_commit(repo, "t_bbb")
    _git("push", "-u", "origin", branch, cwd=wt)
    card = _card(repo, "t_bbb", worktree=wt, branch=branch)

    with pytest.raises(kb.PublicationGateError) as err:
        _complete(card)

    assert [m["check"] for m in err.value.receipt["missing"]] == ["merged_pr"]
    assert err.value.receipt["published_sha"]


def test_branch_with_unrelated_history_is_still_scrutinised(repo):
    """No merge base with the trunk is git saying "this branch is its own work".

    ``git merge-base`` exits 1 when the histories are unrelated. Treating that as
    "cannot tell" would silently skip the gate for exactly the branches that least
    resemble the trunk, so it must be read as: there is something to publish.
    """
    wt = repo["root"] / "wt-orphan"
    _git("worktree", "add", "--detach", str(wt), cwd=repo["work"])
    _git("checkout", "--orphan", "wt/orphan", cwd=wt)
    (wt / "only.txt").write_text("unrelated\n")
    _git("add", "-A", cwd=wt)
    _git("commit", "-m", "orphan work", cwd=wt)
    card = _card(repo, "t_orph", worktree=wt, branch="wt/orphan")

    with pytest.raises(kb.PublicationGateError) as err:
        _complete(card)

    assert [m["check"] for m in err.value.receipt["missing"]] == ["branch_published",
                                                                 "merged_pr"]


def test_published_branch_with_a_merged_pr_closes(repo, monkeypatch):
    """The happy path that must never regress: published + merged closes normally."""
    wt, branch = _worktree_with_commit(repo, "t_ccc")
    _git("push", "-u", "origin", branch, cwd=wt)
    monkeypatch.setenv("FAKE_PRS", json.dumps([
        {"number": 7, "state": "MERGED", "headRefName": branch, "title": "land it",
         "url": "https://github.com/acme/repo/pull/7"},
    ]))
    card = _card(repo, "t_ccc", worktree=wt, branch=branch)

    assert _complete(card) is True
    assert _status(card) == "done"
    assert not _events(card, "publication_gate_blocked")


def test_unmerged_pr_in_the_payload_is_not_publication(repo, monkeypatch):
    """``--state merged`` filters server-side; if a payload does say OPEN, it must not count.

    Cheap insurance: a stubbed ``gh``, a proxy, or a future release that ignores
    ``--state`` must not be able to turn an open pull request into a landed one.
    """
    wt, branch = _worktree_with_commit(repo, "t_bbb2")
    _git("push", "-u", "origin", branch, cwd=wt)
    monkeypatch.setenv("FAKE_PRS", json.dumps([
        {"number": 11, "state": "OPEN", "headRefName": branch, "title": "not landed yet",
         "url": "https://github.com/acme/repo/pull/11"},
    ]))
    card = _card(repo, "t_bbb2", worktree=wt, branch=branch)

    with pytest.raises(kb.PublicationGateError) as err:
        _complete(card)

    assert [m["check"] for m in err.value.receipt["missing"]] == ["merged_pr"]


def test_merged_pr_matched_by_task_id_in_title_closes(repo, monkeypatch):
    """A squash-merged PR deletes its head branch; the card id in the title is the fallback."""
    wt, branch = _worktree_with_commit(repo, "t_ddd")
    _git("push", "-u", "origin", branch, cwd=wt)
    card = _card(repo, "t_ddd", worktree=wt, branch=branch)
    monkeypatch.setenv("FAKE_PRS", json.dumps([
        {"number": 8, "headRefName": "feature/gone", "title": f"{card}: land the work",
         "url": "https://github.com/acme/repo/pull/8"},
    ]))

    assert _complete(card) is True


def test_scratch_card_is_out_of_scope(repo):
    """Structural scope: infrastructure/docs/rescue cards are not branch-backed."""
    card = _card(repo, "t_eee", kind="scratch")
    assert _complete(card) is True


def test_declared_pr_contract_stays_with_the_exact_head_gate(repo):
    """A card declaring OWNER/REPO belongs to kanban_pr_acceptance, not to this gate.

    Whether such a card may close is that gate's call — the assertion here is only
    that this one stays out of it: no verdict, no refusal, no event.
    """
    from hermes_cli.kanban_publication_gate import evaluate_publication_gate

    wt, branch = _worktree_with_commit(repo, "t_fff")
    card = _card(repo, "t_fff", worktree=wt, branch=branch, contract="acme/repo")
    with connect() as conn:
        assert evaluate_publication_gate(conn, card) is None

    try:
        _complete(card)
    except kb.PublicationGateError as exc:
        pytest.fail(f"a declared contract is not this gate's business: {exc}")
    assert not _events(card, "publication_gate_blocked")


def test_worktree_without_new_commits_is_not_gated(repo):
    """An empty worktree produced nothing to publish — refusing it would be a false positive."""
    branch = "wt/t_ggg"
    wt = repo["root"] / "wt-t_ggg"
    _git("worktree", "add", "-b", branch, str(wt), cwd=repo["work"])
    card = _card(repo, "t_ggg", worktree=wt, branch=branch)
    assert _complete(card) is True


def test_explicit_exemption_is_audited_and_allows_the_close(repo):
    """A legitimate non-branch close bypasses the gate, on the record and only once asked."""
    from hermes_cli.kanban_publication_gate import exempt_publication_gate, is_exempt

    wt, branch = _worktree_with_commit(repo, "t_hhh")
    card = _card(repo, "t_hhh", worktree=wt, branch=branch)
    with pytest.raises(kb.PublicationGateError):
        _complete(card)

    with connect() as conn:
        with pytest.raises(ValueError):
            # A blank reason is refused: the silent bypass is what the gate exists to stop.
            exempt_publication_gate(conn, card, reason="   ")
        exempt_publication_gate(conn, card, reason="docs-only change, no branch", actor="vic")

    exemption = _events(card, "publication_gate_exempt")
    assert len(exemption) == 1
    payload = json.loads(exemption[0]["payload"])
    assert payload["actor"] == "vic"
    assert payload["reason"] == "docs-only change, no branch"

    with connect() as conn:
        assert is_exempt(conn, card) is True
    assert _complete(card) is True


def test_force_override_closes_but_stays_on_the_record(repo):
    """An operator can always close a card — and the override is written down."""
    wt, branch = _worktree_with_commit(repo, "t_iii")
    card = _card(repo, "t_iii", worktree=wt, branch=branch)

    assert _complete(card, force=True) is True
    forced = _events(card, "publication_gate_forced")
    assert len(forced) == 1
    payload = json.loads(forced[0]["payload"])
    assert payload["forced"] is True and payload["ok"] is True
    assert payload["missing"], "the override must record what it overrode"
    assert _status(card) == "done"


def test_unreachable_remote_and_gh_fail_open(repo, monkeypatch):
    """A gate must never brick the board: an unknowable state must not block a close."""
    wt, branch = _worktree_with_commit(repo, "t_jjj")
    card = _card(repo, "t_jjj", worktree=wt, branch=branch)
    # Neither the remote nor gh can be asked -> every verdict is "unknown" -> open.
    monkeypatch.setenv("HERMES_KANBAN_PUBLICATION_REMOTE", "no-such-remote")
    monkeypatch.setenv("PATH", str(repo["root"] / "empty-bin"))
    assert _complete(card) is True
    assert not _events(card, "publication_gate_blocked")


def test_kill_switch_disables_the_gate(repo, monkeypatch):
    """Rollback path: the switch turns the gate off without a code change."""
    wt, branch = _worktree_with_commit(repo, "t_kkk")
    card = _card(repo, "t_kkk", worktree=wt, branch=branch)
    monkeypatch.setenv("HERMES_KANBAN_PUBLICATION_GATE", "0")
    assert _complete(card) is True
    assert not _events(card, "publication_gate_blocked")
