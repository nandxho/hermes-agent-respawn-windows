"""Structural publication gate: refuse to close a card whose work never landed.

Root cause of the 2026-10-08 incident: five cards were marked ``done`` while their
commits existed **only in the local ``.git``** — the worker committed in its
worktree and nobody pushed the branch or opened a PR. The worktrees were later
cleaned up and the branches were one ``git gc`` away from disappearing, yet the
board reported the work as shipped. Without a gate the same process hole reopens.

The gate is wired into :func:`hermes_cli.kanban_db.complete_task` as a
pre-transaction gate (same shape as ``_gate_empty_completion``): nothing is
mutated, an auditable event lands, and the caller retries with evidence.

Scope is **structural**, never project-based: only cards whose
``workspace_kind`` is ``worktree`` are checked, because every card on a board may
carry ``project_id=NULL`` (``himex`` does) and project scoping is therefore
unavailable. ``scratch`` and ``dir`` cards are out of scope by construction, so
infrastructure, rescue and documentation tasks — whose destination is not a
branch in ``origin`` — need no silent bypass.

Two checks, exactly as measured sound for this board:

1. **The branch is published** — ``git ls-remote origin refs/heads/<branch>``
   returns a SHA.
2. **A merged pull request names it** — ``gh pr list --repo <repo> --state
   merged`` has ``headRefName == <branch>`` or the card id in its title.

The methods measured as false-positive sources are deliberately **not** used:
``git branch --merged``, ``git branch --contains``, ``git cherry``, blob
comparison against ``origin/main``, or the ``branch_name`` column alone.

Failure direction: a gate must never block the whole board, so an *unknowable*
state (git/gh missing, remote unreachable, branch unreadable) fails **open** —
the card is refused only on positive evidence that the work is local-only. A
card whose branch produced no commits at all is skipped. Legitimate closes whose
destination is not an ``origin`` branch are exempted explicitly and auditably
via :func:`exempt_publication_gate` (a blank reason is refused).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Optional

DEFAULT_REMOTE = "origin"
"""Remote the branch must be published to; a board may override with config."""

WORKTREE_BRANCH_PREFIX = "wt/"
"""Board convention for a worktree card's branch (``wt/<task_id>``)."""

_ENV_SWITCH = "HERMES_KANBAN_PUBLICATION_GATE"
_ENV_REPO = "HERMES_KANBAN_PUBLICATION_REPO"
_SHA = re.compile(r"[0-9a-f]{40}")
_GITHUB_URL = re.compile(r"github\.com[:/]([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?$")
_DEFAULT_BRANCHES = ("main", "master")


# --- Configuration ---------------------------------------------------------


def gate_enabled() -> bool:
    """Whether the publication gate runs. ``HERMES_KANBAN_PUBLICATION_GATE`` wins
    (``0``/``false``/``no``/``off`` disables); otherwise ``kanban.publication_gate``
    (default on). A config read failure fails open — never block a board on a
    config glitch."""
    env = os.environ.get(_ENV_SWITCH)
    if env is not None and env.strip():
        return env.strip().lower() not in {"0", "false", "no", "off"}
    try:
        from hermes_cli.config import load_config_readonly

        kanban = (load_config_readonly() or {}).get("kanban") or {}
        return bool(kanban.get("publication_gate", True))
    except Exception:
        return True


def _configured_remote() -> str:
    env = os.environ.get("HERMES_KANBAN_PUBLICATION_REMOTE", "").strip()
    if env:
        return env
    try:
        from hermes_cli.config import load_config_readonly

        kanban = (load_config_readonly() or {}).get("kanban") or {}
        return (kanban.get("publication_gate_remote") or "").strip() or DEFAULT_REMOTE
    except Exception:
        return DEFAULT_REMOTE


def _configured_repo() -> Optional[str]:
    """Board-wide ``OWNER/REPO`` for the merged-PR check, when set.

    Boards whose cards all target one repository (``himex`` -> the product repo)
    can pin it here; otherwise the slug is read from the worktree's own remote.
    """
    env = os.environ.get(_ENV_REPO, "").strip()
    if env:
        return env
    try:
        from hermes_cli.config import load_config_readonly

        kanban = (load_config_readonly() or {}).get("kanban") or {}
        return (kanban.get("publication_gate_repo") or "").strip() or None
    except Exception:
        return None


# --- Process helpers -------------------------------------------------------


def _run_git(workdir: str, args: list[str], *, timeout: int = 25) -> tuple[Optional[int], str]:
    """``git -C workdir args`` as ``(returncode, stdout)``; ``returncode`` is
    ``None`` when git could not be executed at all (missing binary, timeout)."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(workdir), *args], stdin=subprocess.DEVNULL,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None, ""
    return proc.returncode, proc.stdout or ""


def _run_gh(workdir: str, args: list[str], profile_home: Optional[str], *, timeout: int = 30) -> tuple[Optional[int], str]:
    """``gh args`` in *workdir* as the card's assignee profile; ``returncode`` is
    ``None`` when gh could not be executed."""
    from hermes_cli.kanban_pr_acceptance import _gh_env

    try:
        proc = subprocess.run(
            ["gh", *args], stdin=subprocess.DEVNULL, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, check=False,
            cwd=str(workdir), env=_gh_env(profile_home),
        )
    except (OSError, subprocess.SubprocessError):
        return None, ""
    return proc.returncode, proc.stdout or ""


def _assignee_home(assignee: Optional[str]) -> Optional[str]:
    """The assignee profile's home so ``gh`` runs as that identity; ``None`` for an
    unassigned card (ambient login). An unresolvable profile is treated as
    unknowable, never as a silent fall-through."""
    if not assignee:
        return None
    try:
        from hermes_cli.kanban_pr_acceptance import _assignee_profile_home

        return _assignee_profile_home(assignee)
    except Exception:
        return None


# --- Exemption -------------------------------------------------------------


def is_exempt(conn, task_id: str) -> bool:
    """Whether the card carries an explicit, auditable exemption."""
    try:
        row = conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'publication_gate_exempt' LIMIT 1",
            (task_id,),
        ).fetchone()
    except Exception:
        return False
    return row is not None


def exempt_publication_gate(conn, task_id: str, *, reason: str, actor: Optional[str] = None) -> None:
    """Record an explicit exemption for *task_id* on the task event log.

    A blank reason is refused: a silent bypass is exactly what this gate exists
    to prevent. The record names the actor so a reviewer can ask why.
    """
    reason = (reason or "").strip()
    if not reason:
        raise ValueError("publication gate exemption requires a non-empty reason")
    from hermes_cli.kanban_db import _append_event
    from hermes_cli.kanban_db_connect import write_txn

    with write_txn(conn):
        _append_event(conn, task_id, "publication_gate_exempt", {
            "gate": "publication", "reason": reason[:500], "actor": actor,
        })


# --- Checks ----------------------------------------------------------------


def _resolve_branch(ws: str, recorded: Optional[str], task_id: str) -> Optional[str]:
    """The branch the card's work lives on: the worktree's own HEAD when readable
    (authoritative), else the recorded column, else the ``wt/<id>`` convention."""
    rc, out = _run_git(ws, ["symbolic-ref", "--short", "HEAD"], timeout=10)
    if rc == 0:
        current = (out or "").strip()
        if current:
            return current
    recorded = (recorded or "").strip()
    if recorded:
        return recorded
    if task_id:
        return f"{WORKTREE_BRANCH_PREFIX}{task_id}"
    return None


def _default_remote_ref(ws: str, remote: str) -> Optional[str]:
    """``refs/remotes/<remote>/<default>`` — the trunk the work would land on."""
    for name in _DEFAULT_BRANCHES:
        ref = f"refs/remotes/{remote}/{name}"
        rc, _ = _run_git(ws, ["rev-parse", "--verify", "--quiet", ref], timeout=10)
        if rc == 0:
            return ref
    return None


def _branch_produced_nothing(ws: str, branch: str, remote: str) -> Optional[bool]:
    """``True`` when the branch holds no commit beyond its merge base with the
    remote trunk (nothing to publish), ``None`` when git cannot tell.

    Only a definitive ``True`` skips the gate: an empty worktree card is not a
    hollow close, but an unreadable one must not be blocked either.
    """
    base_ref = _default_remote_ref(ws, remote)
    if base_ref is None:
        return None
    branch_ref = f"refs/heads/{branch}"
    rc, out = _run_git(ws, ["merge-base", branch_ref, base_ref], timeout=15)
    if rc == 1:
        # git answered: the two histories have no common ancestor. The branch is not a
        # child of the remote trunk, so it certainly carries work of its own — that is
        # an answer, not an inability to tell, and must not skip the gate.
        return False
    if rc != 0:
        return None
    base = (out or "").strip().splitlines()
    if not base or not _SHA.fullmatch(base[0].strip()[:40]):
        return None
    rc, out = _run_git(ws, ["rev-list", "--count", f"{base[0].strip()}..{branch_ref}"], timeout=15)
    if rc != 0:
        return None
    try:
        return int((out or "0").strip()) == 0
    except ValueError:
        return None


def _published_sha(ws: str, remote: str, branch: str) -> tuple[Optional[str], str]:
    """``(sha|None, verdict)`` with verdict ``published``/``absent``/``unknown``.

    ``absent`` is definitive (the remote answered and has no such ref);
    ``unknown`` means the remote could not be asked at all.
    """
    rc, out = _run_git(ws, ["ls-remote", "--heads", remote, f"refs/heads/{branch}"], timeout=25)
    if rc is None or rc != 0:
        return None, "unknown"
    for line in (out or "").splitlines():
        sha = line.strip().split()[0] if line.strip() else ""
        if _SHA.fullmatch(sha):
            return sha, "published"
    return None, "absent"


def _repo_slug(ws: str, remote: str) -> Optional[str]:
    """``OWNER/REPO`` for the merged-PR check: board config first, else the
    worktree's own remote URL."""
    configured = _configured_repo()
    if configured:
        return configured
    rc, out = _run_git(ws, ["remote", "get-url", remote], timeout=10)
    if rc != 0 or not out:
        return None
    match = _GITHUB_URL.search(out.strip())
    return match[1] if match else None


def _merged_pr(ws: str, repo: Optional[str], branch: str, task_id: str,
               profile_home: Optional[str]) -> tuple[Optional[dict], str]:
    """``(pr|None, verdict)`` with verdict ``merged``/``absent``/``unknown``.

    A PR counts when its head branch is the card's branch, or — as the incident
    card allows — the task id appears in its title (a squash-merged PR whose
    branch was deleted still carries both).
    """
    if not repo:
        return None, "unknown"
    rc, out = _run_gh(
        ws, ["pr", "list", "--repo", repo, "--state", "merged", "--limit", "200",
             "--json", "number,headRefName,title,url,state"],
        profile_home,
    )
    if rc is None or rc != 0:
        return None, "unknown"
    try:
        prs = json.loads(out or "[]")
    except ValueError:
        return None, "unknown"
    if not isinstance(prs, list):
        return None, "unknown"
    for pr in prs:
        if not isinstance(pr, dict):
            continue
        # gh is asked for merged PRs only, but a payload that does carry a state must
        # still say MERGED: a stub, a proxy, or a gh that ignored --state must not let
        # an open pull request pass as publication.
        if str(pr.get("state") or "MERGED").strip().upper() != "MERGED":
            continue
        head = str(pr.get("headRefName") or "").strip()
        title = str(pr.get("title") or "")
        if head == branch or (task_id and task_id in title):
            return {"number": pr.get("number"), "url": pr.get("url"), "head_ref": head}, "merged"
    return None, "absent"


# --- Gate ------------------------------------------------------------------


def evaluate_publication_gate(conn, task_id: str, *, force: bool = False) -> Optional[dict]:
    """Decide the publication gate for *task_id*.

    ``None`` means the gate does not apply (not a worktree card, a declared PR
    contract that the exact-head gate already owns, already exempted, empty
    branch, unreadable evidence, or nothing missing). A receipt means act on it:
    ``ok`` decides, ``missing`` lists what is absent with its remedy, and
    ``forced`` marks an operator override that must still be audited.
    """
    if not gate_enabled():
        return None
    row = conn.execute(
        "SELECT workspace_kind, workspace_path, branch_name, completion_contract, assignee "
        "FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        return None
    if (row["workspace_kind"] or "") != "worktree":
        return None
    contract = (row["completion_contract"] or "").strip()
    if contract and contract != "local-only":
        return None  # a declared OWNER/REPO or PR contract is the exact-head gate's job
    if is_exempt(conn, task_id):
        return None

    ws = (row["workspace_path"] or "").strip()
    if not ws or not Path(ws).is_dir():
        return None  # nothing to inspect (scratch card, or workspace already cleaned)

    remote = _configured_remote()
    branch = _resolve_branch(ws, row["branch_name"], task_id)
    if not branch or branch in _DEFAULT_BRANCHES or branch == "HEAD":
        return None
    if _branch_produced_nothing(ws, branch, remote) is not False:
        return None  # empty branch, or git could not tell: never block on unknowable state

    published_sha, pub_verdict = _published_sha(ws, remote, branch)
    repo = _repo_slug(ws, remote)
    pr, pr_verdict = _merged_pr(ws, repo, branch, task_id, _assignee_home(row["assignee"]))

    missing: list[dict[str, str]] = []
    if pub_verdict == "absent":
        missing.append({
            "check": "branch_published",
            "detail": f"{remote} has no refs/heads/{branch}",
            "fix": f"git -C {ws} push -u {remote} {branch}",
        })
    if pr_verdict == "absent":
        missing.append({
            "check": "merged_pr",
            "detail": f"no merged pull request on {repo or 'the card repository'} names {branch}",
            "fix": (f"gh pr create --repo {repo or 'OWNER/REPO'} --head {branch} --fill && "
                    f"gh pr merge --repo {repo or 'OWNER/REPO'} --squash {branch}"),
        })
    if not missing:
        return None  # landed, or every unverifiable check failed open

    receipt: dict[str, Any] = {
        "gate": "publication",
        "ok": False,
        "branch": branch, "remote": remote, "repo": repo,
        "published_sha": published_sha,
        "pr_url": pr.get("url") if pr else None,
        "missing": missing,
        "forced": bool(force),
    }
    if force:
        receipt.update(
            ok=True, forced=True,
            reason="operator override (force); exemption recorded for audit",
            missing=missing,
        )
        return receipt
    receipt["remedy"] = (
        "Publish the branch and merge its pull request, then retry the completion. "
        "If this card's output is not a branch in " + remote + " (infrastructure, "
        "rescue, documentation), exempt it explicitly with "
        f"exempt_publication_gate(conn, {task_id!r}, reason=...) so the bypass is audited."
    )
    return receipt


def describe_missing(receipt: dict) -> str:
    """One-line, actionable description of what the gate found absent."""
    parts = [f"{m['check']}: {m['detail']} (fix: {m['fix']})" for m in receipt.get("missing", [])]
    return "; ".join(parts) or "publication evidence unavailable"
