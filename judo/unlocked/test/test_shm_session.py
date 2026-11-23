# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Session cleanup must never unlink queues used by a live runner or forked worker."""

import _posixshmem
import multiprocessing as mp
import os
import sys
import unittest
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from judo.unlocked.output_stage import OutputStage
from judo.unlocked.shm_session import (
    cleanup_abandoned_sessions,
    queue_frame_name,
    register_segment,
    session_queue_name,
    shared_memory_session,
)


def _hold_real_queue(root: Path | None, ready: Connection) -> None:
    with shared_memory_session(root):
        stage = OutputStage("sim_bard_new_random_namespace__camera", bytes)
        stage.open()
        stage.push((b"camera-frame" * 1024,))
        ready.send(stage[0].name)
        ready.recv()
        stage.close()


class _SessionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ctx = mp.get_context("fork")

    def _start(self, target: Callable[..., None], *args: object) -> tuple[BaseProcess, Connection]:
        receive, send = self.ctx.Pipe()
        process = self.ctx.Process(target=target, args=(*args, send))
        process.start()
        send.close()

        def stop() -> None:
            if process.is_alive():
                process.kill()
            process.join(timeout=10)
            receive.close()

        self.addCleanup(stop)
        self.assertTrue(receive.poll(10), "child failed to publish readiness")
        return process, receive

    def _exists(self, name: str) -> bool:
        try:
            segment = SharedMemory(name=name)
        except FileNotFoundError:
            return False
        segment.close()
        return True


def _create_segment(queue: str, name: str) -> None:
    register_segment(queue, name)
    SharedMemory(name=name, create=True, size=8).close()


def _worker_creates_frame(queue: str, ready: Connection) -> None:
    ready.send(True)
    ready.recv()  # Parent leaves the session before this worker allocates its frame.
    name = queue_frame_name(queue, 2**63)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(_create_segment, queue, name).result()
    ready.send(name)
    ready.recv()


@unittest.skipUnless(sys.platform in ("linux", "darwin"), "POSIX shared-memory ownership")
class TestSharedMemorySession(_SessionTestCase):
    """Run the same production strategy with real POSIX segments on Linux and macOS CI."""

    def setUp(self) -> None:
        super().setUp()
        self.addCleanup(cleanup_abandoned_sessions, self.root)

    def test_default_registry_is_persistent_and_private(self) -> None:
        with patch.object(Path, "home", return_value=self.root):
            with shared_memory_session():
                state_path = "Library/Application Support" if sys.platform == "darwin" else ".local/state"
                registry = self.root / state_path / "starfish" / f"starfish-sessions-{os.getuid()}"
                self.assertTrue(registry.is_dir())
                self.assertEqual(registry.stat().st_mode & 0o777, 0o700)
                self.assertEqual(len(list(registry.glob("*.segments"))), 1)
                name = session_queue_name("a" * 20 + "_0")
                _create_segment(name, name)
                self.assertEqual(cleanup_abandoned_sessions(), 0)
            self.assertFalse(self._exists(name))
            self.assertEqual(cleanup_abandoned_sessions(), 0)

    def test_names_are_bounded_and_nested_sessions_are_isolated(self) -> None:
        with shared_memory_session(self.root):
            name = session_queue_name("a" * 20 + "_" + str(2**63))
            _create_segment(name, name)
            frames = [queue_frame_name(name, i) for i in (0, 1, 2**63, 10**100)]
            self.assertEqual(len(set(frames)), 4)
            for frame in frames:
                self.assertLessEqual(len("/" + frame), 30)
                _create_segment(name, frame)
            self.assertLessEqual(len("/" + name), 30)
            with shared_memory_session(self.root):
                inner = session_queue_name("a" * 20 + "_" + str(2**63))
                self.assertNotEqual(name, inner)
                _create_segment(inner, inner)
            self.assertFalse(self._exists(inner))
            self.assertTrue(self._exists(name))
            self.assertEqual(session_queue_name("a" * 20 + "_" + str(2**63)), name)
            self.assertEqual(cleanup_abandoned_sessions(self.root), 0)
        self.assertFalse(self._exists(name))
        self.assertTrue(all(not self._exists(frame) for frame in frames))
        self.assertEqual(session_queue_name("unowned"), "unowned")

    def test_real_queues_survive_live_cleanup_and_are_reclaimed_after_sigkill(self) -> None:
        live, live_ready = self._start(_hold_real_queue, self.root)
        live_name = live_ready.recv()
        crashed, crashed_ready = self._start(_hold_real_queue, self.root)
        crashed_name = crashed_ready.recv()
        self.assertEqual(cleanup_abandoned_sessions(self.root), 0)
        for name in (live_name, crashed_name):
            self.assertTrue(self._exists(name))
            self.assertTrue(self._exists(queue_frame_name(name, 0)))
        crashed.kill()
        crashed.join(timeout=10)
        self.assertFalse(crashed.is_alive())
        self.assertEqual(cleanup_abandoned_sessions(self.root), 2)
        self.assertFalse(self._exists(crashed_name))
        self.assertFalse(self._exists(queue_frame_name(crashed_name, 0)))
        self.assertTrue(live.is_alive())
        self.assertTrue(self._exists(live_name))
        self.assertTrue(self._exists(queue_frame_name(live_name, 0)))

    def test_inherited_worker_can_record_from_thread_after_parent_exit(self) -> None:
        with shared_memory_session(self.root):
            queue = session_queue_name("a" * 20 + "_0")
            _create_segment(queue, queue)
            worker, ready = self._start(_worker_creates_frame, queue)
            self.assertTrue(ready.recv())
        ready.send(True)
        self.assertTrue(ready.poll(10))
        frame = ready.recv()
        self.assertEqual(cleanup_abandoned_sessions(self.root), 0)
        self.assertTrue(self._exists(frame))
        worker.kill()
        worker.join(timeout=10)
        self.assertEqual(cleanup_abandoned_sessions(self.root), 2)
        self.assertFalse(self._exists(queue))
        self.assertFalse(self._exists(frame))

    def test_concurrent_threads_record_all_allocations(self) -> None:
        with shared_memory_session(self.root):
            queue = session_queue_name("a" * 20 + "_0")
            names = [queue_frame_name(queue, i) for i in range(64)]
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda name: _create_segment(queue, name), names))
            self.assertTrue(all(self._exists(name) for name in names))
        self.assertTrue(all(not self._exists(name) for name in names))

    def test_queue_consumer_uses_the_same_compact_frame_names(self) -> None:
        with shared_memory_session(self.root):
            stage = OutputStage("session_camera", bytes)
            view = stage[0].make_view()
            stage.open()
            view.open()
            try:
                # Allocation happens in a thread without the materializer's ContextVar state.
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(stage.push, (b"camera-frame",)).result()
                self.assertEqual(view[0], b"camera-frame")
                manifest = next(self.root.glob("*/*.segments"))
                self.assertEqual(manifest.read_text().splitlines(), [stage[0].name, queue_frame_name(stage[0].name, 0)])
            finally:
                view.close()
                stage.close()

    def test_registration_precedes_creation_and_failed_journal_never_allocates(self) -> None:
        with shared_memory_session(self.root):
            queue = session_queue_name("a" * 20 + "_0")
            name = queue_frame_name(queue, 0)
            with (
                patch("judo.unlocked.shm_session.os.write", return_value=1),
                self.assertRaisesRegex(OSError, "Incomplete"),
            ):
                _create_segment(queue, name)
            self.assertFalse(self._exists(name))
            register_segment(queue, name)
            self.assertFalse(self._exists(name))

    def test_unowned_segment_is_never_reclaimed(self) -> None:
        unowned = SharedMemory(create=True, size=8)
        try:
            with shared_memory_session(self.root):
                name = session_queue_name("a" * 20 + "_0")
                _create_segment(name, name)
            self.assertFalse(self._exists(name))
            self.assertTrue(self._exists(unowned.name))
        finally:
            unowned.close()
            unowned.unlink()

    def test_missing_and_zero_length_allocations_and_torn_record(self) -> None:
        with shared_memory_session(self.root):
            queue = session_queue_name("a" * 20 + "_0")
            register_segment(queue, queue)  # Crash before shm_open: nothing to unlink.
            zero = queue_frame_name(queue, 0)
            register_segment(queue, zero)
            fd = _posixshmem.shm_open("/" + zero, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.close(fd)  # Crash before ftruncate: attaching with SharedMemory would fail.
            manifest = next(self.root.glob("*/*.segments"))
            with manifest.open("ab") as records:
                records.write(b"sfm_incomplete")
            name = queue_frame_name(queue, 1)
            _create_segment(queue, name)  # Repairs an incomplete previous record before appending.
            with manifest.open("ab") as records:
                records.write(b"sfm_incomplete")
        self.assertFalse(self._exists(name))
        with self.assertRaises(FileNotFoundError):
            _posixshmem.shm_unlink("/" + zero)
        self.assertFalse(manifest.exists())

    def test_unlink_failure_retains_manifest_and_preserves_graph_error(self) -> None:
        with (
            patch("judo.unlocked.shm_session._posixshmem.shm_unlink", side_effect=PermissionError("busy")),
            self.assertLogs("judo.unlocked.shm_session", level="WARNING"),
            self.assertRaisesRegex(RuntimeError, "graph failure"),
        ):
            with shared_memory_session(self.root):
                queue = session_queue_name("a" * 20 + "_0")
                _create_segment(queue, queue)
                raise RuntimeError("graph failure")
        self.assertTrue(self._exists(queue))
        self.assertEqual(len(list(self.root.glob("*/*.segments"))), 1)
        self.assertEqual(cleanup_abandoned_sessions(self.root), 1)
        self.assertFalse(self._exists(queue))
        self.assertFalse(list(self.root.glob("*/*.segments")))

    def test_cleanup_failure_does_not_mask_graph_error_and_can_retry(self) -> None:
        with (
            patch("judo.unlocked.shm_session._reclaim", side_effect=PermissionError("unavailable")),
            self.assertLogs("judo.unlocked.shm_session", level="WARNING"),
            self.assertRaisesRegex(RuntimeError, "graph failure"),
        ):
            with shared_memory_session(self.root):
                name = session_queue_name("a" * 20 + "_0")
                _create_segment(name, name)
                raise RuntimeError("graph failure")
        self.assertTrue(self._exists(name))
        self.assertEqual(cleanup_abandoned_sessions(self.root), 1)
        self.assertFalse(self._exists(name))

    def test_partial_unlink_failure_retains_lease_for_retry(self) -> None:
        unlink = _posixshmem.shm_unlink

        def fail_frame(name: str) -> None:
            if name == "/" + frame:
                raise PermissionError("frame busy")
            unlink(name)

        with (
            patch("judo.unlocked.shm_session._posixshmem.shm_unlink", side_effect=fail_frame),
            self.assertLogs("judo.unlocked.shm_session", level="WARNING"),
        ):
            with shared_memory_session(self.root):
                index = session_queue_name("a" * 20 + "_0")
                frame = queue_frame_name(index, 0)
                _create_segment(index, index)
                _create_segment(index, frame)
        self.assertFalse(self._exists(index))
        self.assertTrue(self._exists(frame))
        self.assertEqual(len(list(self.root.glob("*/[0-9a-f]*.lock"))), 1)
        self.assertEqual(cleanup_abandoned_sessions(self.root), 1)
        self.assertFalse(self._exists(frame))
        self.assertFalse(list(self.root.glob("*/[0-9a-f]*.lock")))

    def test_insecure_registry_is_rejected(self) -> None:
        registry = self.root / f"starfish-sessions-{os.getuid()}"
        registry.mkdir(mode=0o755)
        registry.chmod(0o755)
        # Restore the permissions before the common teardown reclaims any abandoned sessions.
        self.addCleanup(registry.chmod, 0o700)
        with self.assertRaises(PermissionError):
            cleanup_abandoned_sessions(self.root)

    def test_unpublished_manifest_does_not_require_directory_scanning(self) -> None:
        with (
            patch.object(Path, "touch", side_effect=OSError("creation failed")),
            self.assertRaisesRegex(OSError, "creation failed"),
        ):
            with shared_memory_session(self.root):
                self.fail("Session must not start without its manifest")
        self.assertEqual(len(list(self.root.glob("*/[0-9a-f]*.lock"))), 1)
        self.assertEqual(cleanup_abandoned_sessions(self.root), 0)
        self.assertFalse(list(self.root.glob("*/[0-9a-f]*.lock")))


if __name__ == "__main__":
    unittest.main()
