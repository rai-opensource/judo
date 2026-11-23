# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Hermetic tests for graph construction, serialization, dry runs, and shutdown reporting.

These exercise the exact units added for the Hydra migration -- :func:`_instantiate_kwargs` (nested
``_target_`` / ``_partial_`` construction, with plain-primitive kwargs passing through untouched)
and :meth:`GraphBuilder.to_yaml` / :meth:`GraphBuilder.from_yaml` -- using only stdlib ``_target_``
callables, so they need neither the heavy ``module.*`` runtime classes nor any DVC assets. The full
``spot_control`` end-to-end path (which builds the real node classes) lives in ``//deploy/app/test:test_graph``.
"""

from __future__ import annotations

import collections
import functools
import multiprocessing as mp
import subprocess
import sys
import textwrap
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from judo.unlocked.graph.builder import GraphBuilder, _instantiate_kwargs
from judo.unlocked.graph.runner import _run_post_run_hooks, run_builder
from judo.unlocked.node import Node, NodeStop
from judo.unlocked.schedule import Threaded
from judo.unlocked.shm_session import register_segment, session_queue_name, shared_memory_session


def test_plain_kwargs_pass_through_untouched() -> None:
    """Primitive-only kwargs (every pre-migration instance) are returned unchanged -- no Hydra call."""
    kwargs = {"robot": "mujoco", "n": 3, "flag": True, "nested": {"a": "dexterity.tasks.Missing"}}
    out = _instantiate_kwargs(kwargs)
    assert out == kwargs
    assert out is not kwargs  # a copy, so callers can't mutate the stored spec


def test_nested_target_is_instantiated() -> None:
    """A nested ``{_target_: ...}`` sub-config is built into a live object by Hydra."""
    out = _instantiate_kwargs({"counter": {"_target_": "collections.Counter", "b": 2}})
    assert isinstance(out["counter"], collections.Counter)
    assert out["counter"]["b"] == 2


def test_partial_target_yields_bound_callable() -> None:
    """``_partial_: true`` yields a ``functools.partial`` -- the shape the robot_provider relies on."""
    out = _instantiate_kwargs({"make": {"_target_": "collections.Counter", "_partial_": True, "b": 2}})
    assert isinstance(out["make"], functools.partial)
    assert out["make"]() == collections.Counter(b=2)


def test_instance_with_target_kwarg_round_trips_through_yaml() -> None:
    """A graph whose instance kwargs carry a nested ``_target_`` round-trips through YAML verbatim."""
    g = GraphBuilder(name="toy")
    g.instance(
        "c",
        collections.Counter,
        data={"_target_": "collections.OrderedDict", "x": 1},
    )
    spec = g.to_dict()
    assert GraphBuilder.from_yaml(g.to_yaml()).to_dict() == spec
    # The nested _target_ survives serialization untouched (it is built only at materialize time).
    inst = next(i for i in spec["instances"] if i["id"] == "c")
    assert inst["kwargs"]["data"]["_target_"] == "collections.OrderedDict"


def test_graph_import_does_not_import_deploy() -> None:
    # unLocked is deployment middleware (it lives under ``judo.unlocked``), so importing the graph
    # builder necessarily pulls in ``deploy`` and ``judo.unlocked.*``. The invariant this guards is
    # that the framework stays isolated from the heavy deployment *application* layer -- no other
    # ``deploy.*`` (e.g. ``deploy.app``/``deploy.nodes``) and no ``dexterity`` modules get imported.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util, sys; "
            "from judo.unlocked.graph.builder import _instantiate_kwargs; "
            "_instantiate_kwargs({'robot': 'mujoco', 'n': 3}); "
            "bad = [n for n in sys.modules "
            "if n.split('.')[0] == 'dexterity' "
            "or (n.split('.')[0] == 'deploy' and n != 'deploy' and not n.startswith('judo.unlocked'))]; "
            "assert not bad, bad",
        ],
        check=True,
    )


def _stop_node() -> None:
    raise NodeStop()


@pytest.mark.parametrize("second_signal", ["SIGINT", "SIGTERM"])
def test_graph_signal_reentry_does_not_deadlock(second_signal: str) -> None:
    """Reenter the installed handler under Event.set's lock, not a timing-dependent burst.

    The subprocess runs the real GraphBuilder.run with in-process scheduler doubles:
    a deadlocked regression cannot leak graph grandchildren or shared memory.
    """
    source = textwrap.dedent(
        """
        import faulthandler
        import signal
        import sys
        import threading
        from unittest.mock import patch
        from judo.unlocked.graph.builder import GraphBuilder

        second = getattr(signal, sys.argv[1])
        nested = False
        calls = []

        def trace(frame, event, arg):
            global nested
            locked_set = (
                frame.f_code is threading.Event.set.__code__ and event == 'line'
                and frame.f_locals['self']._cond._lock.locked()
            )
            handler_return = frame.f_code is signal.getsignal(signal.SIGINT).__code__ and event == 'return'
            if not nested and (locked_set or handler_return):
                nested = True
                print('reentry: ' + ('Event.set lock held' if locked_set else 'handler return'), flush=True)
                signal.raise_signal(second)
            return trace

        class Scheduler:
            name = 'worker'
            alive = True
            exitcode = None
            shutdown_clean = False

            def start(self):
                faulthandler.dump_traceback_later(2)
                sys.settrace(trace)
                signal.raise_signal(signal.SIGINT)
                sys.settrace(None)
                assert nested

            def stop(self):
                calls.append('stop')
                for _ in range(20):
                    signal.raise_signal(signal.SIGINT)
                    signal.raise_signal(signal.SIGTERM)

            def join(self, timeout):
                assert timeout > 0
                calls.append('join')
                signal.raise_signal(second)
                self.alive = False
                self.exitcode = 0
                self.shutdown_clean = True

        graph = GraphBuilder()
        with patch.object(graph, 'build_processes', return_value=([Scheduler()], {})):
            graph.run()
        assert calls == ['stop', 'join'], calls
        assert graph.shutdown_clean
        faulthandler.cancel_dump_traceback_later()
        print('clean shutdown', flush=True)
    """
    )
    # subprocess.run kills AND reaps this child on timeout; no descendants exist.
    try:
        result = subprocess.run(
            [sys.executable, "-u", "-c", source, second_signal],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"Signal handler stalled: {exc.stdout!r}\n{exc.stderr!r}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "clean shutdown" in result.stdout


def _fail_lifecycle() -> None:
    raise RuntimeError("lifecycle failure")


@pytest.mark.parametrize("phase", [None, "pre_open", "post_close"])
def test_scheduler_shutdown_status_shared_with_parent(phase: str | None) -> None:
    """A real forked scheduler publishes status only after cleanup, including caught hook failures."""
    closed = mp.get_context("fork").Event()
    node = Node(
        "shutdown_test",
        _stop_node,
        pre_open=_fail_lifecycle if phase == "pre_open" else None,
        post_close=_fail_lifecycle if phase == "post_close" else closed.set,
    )
    scheduler = Threaded("shutdown_test", [node])
    assert scheduler.shutdown_clean is False
    scheduler.start()
    try:
        scheduler.join(timeout=10)
        assert not scheduler.alive
        # Node catches lifecycle errors: exitcode alone cannot certify clean shutdown.
        assert scheduler.exitcode == 0
        assert scheduler.shutdown_clean is (phase is None)
        assert closed.is_set() is (phase is None)
    finally:
        if scheduler.alive:
            scheduler.kill()
            scheduler.join(timeout=5)


@pytest.mark.parametrize("phase", ["pre_open", "post_close"])
def test_node_lifecycle_error_is_read_only(phase: str) -> None:
    lifecycle_kwargs: dict[str, Any] = {phase: _fail_lifecycle}
    node = Node("lifecycle_test", _stop_node, **lifecycle_kwargs)
    before = node.lifecycle_error
    assert before is False
    node.open()
    node.close()
    after = node.lifecycle_error
    assert after is True
    with pytest.raises(AttributeError):
        node.lifecycle_error = False  # type: ignore[misc]


def test_node_runtime_errors_are_not_lifecycle_errors() -> None:
    node = Node("runtime_test", _fail_lifecycle)
    node.exec()
    assert node.live
    assert node.lifecycle_error is False


@pytest.mark.parametrize("failure", ["worker", "input_close", "output_close"])
def test_scheduler_incomplete_cleanup_is_not_clean(failure: str) -> None:
    node = Mock(spec=Node, live=False, lifecycle_error=False)
    worker = Mock()
    worker.is_alive.return_value = failure == "worker"
    scheduler = Threaded("incomplete_cleanup", [node])
    scheduler.node_threads = [worker]
    if failure != "worker":
        stage = node.input_stage if failure == "input_close" else node.output_stage
        stage.close.side_effect = RuntimeError("stage cleanup failure")
    with patch.object(scheduler, "_start_node_threads"), patch("judo.unlocked.schedule.gc.freeze"):
        if failure == "worker":
            scheduler.spin()
        else:
            with pytest.raises(RuntimeError, match="stage cleanup failure"):
                scheduler.spin()
    worker.join.assert_called_once_with(timeout=1.0)
    assert scheduler.shutdown_clean is False


@pytest.mark.parametrize("fail_close", [False, True])
def test_run_builder_real_cleanup(fail_close: bool) -> None:
    """Exercise graph aggregation with a real scheduler, without loading any application objects."""
    closed = mp.get_context("fork").Event()
    scheduler = Threaded(
        "worker", [Node("graph_cleanup", _stop_node, post_close=_fail_lifecycle if fail_close else closed.set)]
    )
    g = GraphBuilder()
    with patch.object(g, "build_processes", return_value=([scheduler], {})), patch("signal.signal"):
        try:
            assert run_builder(g, shm_cleanup=False) == int(fail_close)
            assert g.shutdown_clean is (not fail_close)
            assert closed.is_set() is (not fail_close)
        finally:
            if scheduler.alive:
                scheduler.kill()
                scheduler.join(timeout=5)


@pytest.mark.parametrize("clean,exitcode", [(True, 0), (False, 0), (True, 1), (True, -9), (True, None)])
def test_graph_shutdown_requires_clean_status_and_zero_exit(clean: bool, exitcode: int | None) -> None:
    scheduler = Mock(spec=Threaded)
    scheduler.name = "worker"
    scheduler.alive = False
    scheduler.shutdown_clean = clean
    scheduler.exitcode = exitcode
    g = GraphBuilder()
    g.shutdown_clean = True  # A previous run must not leak success into the next run.

    def check_reset() -> tuple[list[Threaded], dict[str, Node]]:
        assert g.shutdown_clean is False
        return [scheduler], {}

    with patch.object(g, "build_processes", side_effect=check_reset), patch("signal.signal"):
        code = run_builder(g, shm_cleanup=False)
    expected = clean and exitcode == 0
    assert g.shutdown_clean is expected
    assert code == (0 if expected else 1)
    scheduler.terminate.assert_not_called()
    scheduler.kill.assert_not_called()


@pytest.mark.parametrize("escalation", ["terminate", "kill", "still_alive"])
def test_graph_escalation_is_unclean_and_kill_is_joined(escalation: str) -> None:
    completed = Mock(spec=Threaded)
    completed.name = "completed"
    completed.alive = False
    completed.exitcode = 0
    completed.shutdown_clean = True
    scheduler = Mock(spec=Threaded)
    scheduler.name = "worker"
    scheduler.alive = True
    scheduler.exitcode = None
    scheduler.shutdown_clean = False

    def terminate() -> None:
        if escalation == "terminate":
            # Even a successful graceful SIGTERM response is still escalation.
            scheduler.alive = False
            scheduler.exitcode = 0
            scheduler.shutdown_clean = True

    def kill() -> None:
        if escalation == "kill":
            scheduler.alive = False
            scheduler.exitcode = -9

    scheduler.terminate.side_effect = terminate
    scheduler.kill.side_effect = kill
    g = GraphBuilder()
    with patch.object(g, "build_processes", return_value=([completed, scheduler], {})), patch("signal.signal"):
        assert run_builder(g, shm_cleanup=False) == 1
    assert g.shutdown_clean is False
    scheduler.terminate.assert_called_once()
    if escalation == "terminate":
        scheduler.kill.assert_not_called()
    else:
        scheduler.kill.assert_called_once()
        assert scheduler.method_calls[-2:] == [("kill", (), {}), ("join", (), {"timeout": 2.0})]


@pytest.mark.parametrize("clean", [False, True, None])
def test_post_run_hook_status_is_opt_in_and_metadata_is_preserved(clean: bool | None) -> None:
    g = GraphBuilder()
    g.add_post_run_hook("shutdown_test_hooks:legacy", label="old")
    g.add_post_run_hook("shutdown_test_hooks:disabled", include_shutdown_status=False)
    g.add_post_run_hook("shutdown_test_hooks:status", include_shutdown_status=True, shutdown_clean="untrusted")
    spec = g.to_dict()
    g = GraphBuilder.from_dict(spec)
    if clean is None:
        del g.shutdown_clean  # Older graph-like callers default to an unclean status.
    else:
        g.shutdown_clean = clean
    hooks = SimpleNamespace(legacy=Mock(return_value=0), disabled=Mock(return_value=0), status=Mock(return_value=0))
    with patch.dict(sys.modules, shutdown_test_hooks=hooks):
        assert _run_post_run_hooks(g, 123.0) == 0
        assert _run_post_run_hooks(g, 123.0) == 0
    hooks.legacy.assert_called_with(since=123.0, label="old")
    hooks.disabled.assert_called_with(since=123.0)
    hooks.status.assert_called_with(since=123.0, shutdown_clean=bool(clean))
    assert g.to_dict() == spec


@pytest.mark.parametrize("clean,hook_code", [(False, 0), (True, 0), (False, 7), (True, 7)])
def test_runner_hooks_receive_failure_and_cannot_mask_it(clean: bool, hook_code: int) -> None:
    g = GraphBuilder()
    g.add_post_run_hook("shutdown_test_hooks:report", include_shutdown_status=True)
    report = Mock(return_value=hook_code)

    def run() -> None:
        g.shutdown_clean = clean  # Model the actual run contract, not just a no-op mock.

    with (
        patch.object(g, "run", side_effect=run),
        patch.dict(sys.modules, shutdown_test_hooks=SimpleNamespace(report=report)),
    ):
        assert run_builder(g, shm_cleanup=False) == (hook_code or (0 if clean else 1))
    assert report.call_args.kwargs["shutdown_clean"] is clean
    assert isinstance(report.call_args.kwargs["since"], float)


def test_runner_noop_run_mock_is_not_success() -> None:
    g = GraphBuilder()
    with patch.object(g, "run"):
        assert run_builder(g, shm_cleanup=False) == 1


@pytest.mark.parametrize("fail", [False, True])
def test_runner_owns_queues_before_runtime_construction(tmp_path: Path, fail: bool) -> None:
    g = GraphBuilder()
    names: list[str] = []

    def run() -> None:
        name = session_queue_name("a" * 20 + "_0")
        assert name.startswith("sfm_")
        register_segment(name, name)
        SharedMemory(name=name, create=True, size=8).close()
        names.append(name)
        if fail:
            raise RuntimeError("construction failed")
        g.shutdown_clean = True

    with (
        patch.object(g, "run", side_effect=run),
        patch(
            "judo.unlocked.graph.runner.shared_memory_session",
            side_effect=lambda: shared_memory_session(tmp_path),
        ),
    ):
        if fail:
            with pytest.raises(RuntimeError, match="construction failed"):
                run_builder(g, shm_cleanup=False)
        else:
            assert run_builder(g, shm_cleanup=False) == 0
    assert names
    for name in names:
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name)


def test_dry_run_does_not_import_or_construct_targets(capsys: pytest.CaptureFixture[str]) -> None:
    g = GraphBuilder.from_dict(
        {
            "name": "declarative",
            "instances": [{"id": "shared", "factory": "nonexistent_hardware:Constructor"}],
            "nodes": [
                {
                    "id": "later",
                    "target": "nonexistent_hardware:Constructor.step",
                    "instance": "shared",
                    "close_order": 2,
                },
                {
                    "id": "first",
                    "target": "nonexistent_model:load",
                    "kwargs": {"path": "missing.onnx"},
                    "close_order": -1,
                },
                {"id": "tie", "target": "nonexistent_model:load", "close_order": -1},
            ],
            "edges": [],
        }
    )
    with (
        patch.object(g, "materialize", side_effect=AssertionError("must not materialize")),
        patch.object(g, "build_processes", side_effect=AssertionError("must not build schedulers")),
        patch("judo.unlocked.graph.builder._import_target", side_effect=AssertionError("must not import")),
        patch("judo.unlocked.graph.builder._instantiate_kwargs", side_effect=AssertionError("must not instantiate")),
        patch("judo.unlocked.graph.runner._cleanup_leftover_shm", side_effect=AssertionError("must not clean shm")),
        patch("judo.unlocked.graph.runner.shared_memory_session", side_effect=AssertionError("must not own shm")),
    ):
        assert run_builder(g, dry_run=True) == 0
    output = capsys.readouterr().out
    assert "graph 'declarative': 1 process(es)" in output
    assert "shutdown tier 0 (grace 3s): main" in output
    assert "process 'main' open/close order: ['first', 'tie', 'later']" in output
    assert g.shutdown_clean is False


def test_dry_run_checks_serialized_ports_without_loading() -> None:
    g = GraphBuilder.from_dict(
        {
            "nodes": [{"id": "node", "target": "nonexistent_hardware:Constructor", "inputs": ["in"]}],
            "edges": [{"from": "node.missing", "to": "node.in"}],
        }
    )
    with pytest.raises(ValueError, match="no output port 'missing'"):
        run_builder(g, dry_run=True)
