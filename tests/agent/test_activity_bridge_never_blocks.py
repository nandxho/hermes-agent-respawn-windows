"""The kanban activity bridge must never block the thread that touches activity.

Report: with a 600s stale threshold the stream watchdog killed at 3019s/3041s in the
same second for several workers, and a wave of gateway "liveness watchdog" aborts landed
on workers whose board heartbeat was under 5s old. Both threads that call
``_touch_activity`` — the stream-monitor poll loop and the worker turn — were parked in
the board write path: the bridge (worker heartbeat + operator-note injection) ran INLINE
on the caller, and a WAL write-txn convoy on the shared dispatch DB holds a writer for the
whole ``kanban_db_connect.write_txn`` busy-timeout chain (6 x 120s). A stalled monitor
cannot evaluate the stale deadline (the kill fires minutes late); a stalled turn clock
stops the gateway liveness watchdog's activity feed (healthy turns get aborted, exit 130,
and are booked ``crashed``).

The invariants below pin the caller free of board I/O, one bridge thread per process
(coalesced, not a thread per touch), and the kill line naming the provider plus the
monitor starvation that made it late.
"""
import logging
import threading
import time

import pytest

import run_agent
from agent import activity_tracking
from agent import chat_completion_helpers as helpers
from tests.waiting import wait_until


def _agent(provider="custom"):
    return run_agent.AIAgent(
        api_key="test-key", base_url="http://127.0.0.1:1/v1", model="m", provider=provider,
        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[], max_iterations=1,
    )


class _BlockingBridge:
    """A board write that never returns until released — the convoy the report measured."""

    def __init__(self):
        self.release = threading.Event()
        self.calls = []
        self.entered = threading.Event()
        self.threads = set()

    def heartbeat(self):
        self.calls.append(time.monotonic())
        self.threads.add(threading.current_thread().name)
        self.entered.set()
        self.release.wait(timeout=8.0)  # bounded so a failing run can never hang teardown

    def inject(self, _agent):
        return None


@pytest.fixture
def bridge(monkeypatch):
    """Bridge both kanban entry points; keep the env that arms the bridge."""
    blocking = _BlockingBridge()
    import tools.kanban_tools as kt
    monkeypatch.setattr(kt, "heartbeat_current_worker_from_env", blocking.heartbeat)
    monkeypatch.setattr(kt, "inject_new_comments_from_env", blocking.inject)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    # Module state: a leaked in-flight flag from another test would make every touch a no-op.
    monkeypatch.setattr(activity_tracking, "_bridge_inflight", False, raising=False)
    yield blocking
    blocking.release.set()
    wait_until(lambda: not activity_tracking._bridge_inflight, timeout=10.0,
               what="the bridge slot to be freed after release")


def test_activity_touch_returns_while_the_board_write_is_blocked(bridge):
    """RED pre-fix: the touch sat in ``heartbeat_current_worker_from_env`` for the whole write."""
    agent = _agent()
    started = time.monotonic()
    agent._touch_activity("waiting for model response (1s elapsed)")
    elapsed = time.monotonic() - started
    assert elapsed < 2.0, (
        f"_touch_activity blocked {elapsed:.1f}s on board I/O — the stale kill and the "
        "gateway liveness clock both ride this call and cannot fire while it is parked")
    wait_until(lambda: bridge.entered.is_set(), timeout=10.0, what="the bridge to reach the board")


def test_touches_coalesce_into_one_bridge_thread(bridge):
    """A slow board must not grow a thread per 30s tick — one in-flight call absorbs them."""
    agent = _agent()
    agent._touch_activity("waiting for model response (1s elapsed)")
    wait_until(lambda: bridge.entered.is_set(), timeout=10.0, what="the first bridge call")
    for _ in range(5):
        agent._touch_activity("waiting for model response (2s elapsed)")
    assert len(bridge.calls) == 1, f"coalescing lost: {len(bridge.calls)} board writes for 6 touches"
    assert bridge.threads == {"kanban-activity-bridge"}, f"bridge ran on {bridge.threads}"


def test_bridge_slot_is_freed_so_heartbeats_resume(bridge):
    """A bridge that never freed its slot would starve ``last_heartbeat_at`` and let the
    board reclaim a LIVE worker (``DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS``)."""
    agent = _agent()
    agent._touch_activity("waiting for model response (1s elapsed)")
    wait_until(lambda: bridge.entered.is_set(), timeout=10.0, what="the first bridge call")
    bridge.release.set()
    wait_until(lambda: not activity_tracking._bridge_inflight, timeout=10.0, what="the slot to free")
    calls_before = len(bridge.calls)
    bridge.release.clear()
    bridge.entered.clear()
    agent._touch_activity("waiting for model response (3s elapsed)")
    wait_until(lambda: bridge.entered.is_set(), timeout=10.0, what="the second bridge call")
    assert len(bridge.calls) == calls_before + 1


# ── the kill line must explain itself ───────────────────────────────────────────


def _stale_call(provider="ollama-cloud", threshold=420.0):
    call = helpers._StreamingCall(_agent(provider=provider), {"model": "deepseek-v4.1-flash"}, None)
    call._stream_stale_timeout = threshold
    call._count_stale_attempt = lambda: None            # breaker bookkeeping is not under test
    call._rearm_or_abandon_after_kill = lambda elapsed: None
    return call


def test_stale_kill_names_the_provider(caplog):
    """The line used to name only the model, so a 420s threshold firing at 3019s could not be
    attributed to a route — the report's first question was "which provider was that?"."""
    call = _stale_call()
    with caplog.at_level(logging.WARNING, logger="agent.chat_completion_helpers"):
        call._kill_stale_stream(420.2)
    text = "\n".join(r.getMessage() for r in caplog.records if "Stream stale for" in r.getMessage())
    assert "provider=ollama-cloud" in text, text
    assert "model=deepseek-v4.1-flash" in text, text
    # Inside the deadline: the poll loop is healthy and must not cry starvation.
    assert "starved" not in text, text


def test_stale_kill_reports_monitor_starvation_when_the_deadline_overshoots(caplog):
    """3019s against a 600s threshold is a blocked monitor thread (0.3s poll), not a slow
    provider — the log said neither, and the delay read as an unexplained 5x."""
    call = _stale_call(threshold=600.0)
    with caplog.at_level(logging.WARNING, logger="agent.chat_completion_helpers"):
        call._kill_stale_stream(3019.0)
    text = "\n".join(r.getMessage() for r in caplog.records if "Stream stale for" in r.getMessage())
    assert "starved 2419s past the deadline" in text, text


# ── acceptance: a real wedged wire aborts inside the budget and names the route ──


def test_wedged_wire_aborts_within_its_threshold_and_names_the_provider(caplog, monkeypatch):
    """The report's criterion 1 against a live socket: a provider that accepts the
    connection and never emits a byte must be killed inside ``stale_timeout_seconds``,
    and the line must carry the route so a late kill can be attributed.

    Uses the ``custom`` route because the budget is resolved at construction from the
    provider record — that resolution (a wrong route id reads the wrong knob) is the
    config half the parent card already fixed; this test pins the code half.
    """
    import threading as _threading
    import time as _time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    release = _threading.Event()

    class _Silent(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            pass

        def do_POST(self):
            n = int(self.headers.get("content-length", 0))
            self.rfile.read(n)
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            release.wait(timeout=30.0)  # accepted, then silent — the wedge from the report

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Silent)
    server.daemon_threads = True
    _threading.Thread(target=lambda: server.serve_forever(poll_interval=0.05), daemon=True).start()
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "2")
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    monkeypatch.setenv("HERMES_STREAM_READ_TIMEOUT", "20")
    try:
        agent = run_agent.AIAgent(
            api_key="test-key", base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
            model="m", provider="custom", platform="cli",  # worker + monitor threads, the gateway shape
            quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[],
            max_iterations=1,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False
        started = _time.monotonic()
        with caplog.at_level(logging.WARNING, logger="agent.chat_completion_helpers"):
            with pytest.raises(Exception):
                agent._interruptible_streaming_api_call(
                    {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        elapsed = _time.monotonic() - started
    finally:
        release.set()
        server.shutdown()
        server.server_close()

    text = "\n".join(r.getMessage() for r in caplog.records if "Stream stale for" in r.getMessage())
    assert "provider=custom" in text, text
    # 2s budget, 0.3s poll: anything near the 20s byte-read timeout means the monitor
    # was starved and the read timeout — not the watchdog — ended the call.
    assert elapsed < 8.0, f"the wedge took {elapsed:.1f}s against a 2s budget — the monitor was starved"

