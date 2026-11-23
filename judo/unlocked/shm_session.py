# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Crash cleanup for graph-owned POSIX shared-memory queues on Linux and macOS.

Shared-memory segments can outlive the processes that created them. Normal queue teardown
unlinks them, but a crash or SIGKILL can bypass that teardown. This module records which
segments belong to each graph run and determines when it is safe to reclaim them; it does
not change how queue messages are transported.

Bookkeeping, not queue data
--------------------------
``_default_root()`` chooses a persistent directory for bookkeeping files. Each session has
a lease file (``<session-id>.lock``) and a manifest (``<session-id>.segments``) listing its
shared-memory names. The queue data itself remains in POSIX shared memory, not this directory.
Only the default bookkeeping location differs by platform; the cleanup strategy is identical.

Session lifecycle
-----------------
1. ``shared_memory_session()`` creates a unique session and exclusively locks its lease.
    Scheduler workers inherit the open descriptor when they fork. Closing the parent's
    descriptor does not release the lock while an inherited worker descriptor remains open.
    Never explicitly unlock the lease: that would also release the workers' shared lock.
2. Before creating each queue index or frame, ``register_segment()`` records its exact name
    in the manifest. Recording BEFORE allocation is essential: a crash between those steps
    leaves a harmless record for a nonexistent segment. Reversing the order could leave an
    allocated segment with no record. A separate manifest lock serializes concurrent writers.
3. At startup, ``cleanup_abandoned_sessions()`` tries to acquire each session's lease lock
    without waiting. A locked lease means a parent or worker still holds it, so cleanup skips
    the session. An available lease allows cleanup to unlink the recorded segments by name.
    Session exit also attempts this cleanup after closing its own lease descriptor.

For example, if app A's parent crashes but a camera worker survives, app B's startup cleanup
cannot acquire A's lease and leaves its queues alone. Once all inherited holders exit, the OS
releases the lock and a later cleanup can reclaim A's segments. No PID checks, heartbeats, or
process-age guesses are needed. All session users must follow this inherited-lease protocol;
an unrelated process merely attaching to shared memory does not acquire a lease automatically.

Reclamation and limits
----------------------
``shm_unlink`` removes a segment's name; existing mappings remain valid until released.
Missing segments are harmless, and unlink failures retain the manifest and lease for retry.
Unowned segments are never guessed at or deleted. Manifests avoid scanning ``/dev/shm``, which
macOS does not expose, and also allow cleanup of allocations that crashed before being sized.

There is no background cleanup service. If the entire app is killed, its records and any
remaining named segments await a later cleanup invocation. Manifests are append-only during
a session, so bookkeeping grows with segment allocations until the session is reclaimed.

In short: the manifest records what belongs to a run; the lease determines when cleanup is safe.
"""

import _posixshmem
import fcntl
import hashlib
import logging
import os
import re
import stat
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_LOG = logging.getLogger(__name__)
_MANIFEST: ContextVar[Path | None] = ContextVar("shm_session_manifest", default=None)
# Queue ownership is inherited across fork and readable by worker threads (unlike ContextVars).
_QUEUE_MANIFESTS: dict[str, Path] = {}
_SEGMENT_NAME = re.compile(r"sfm_[0-9a-f]{24}")
_RECORD_SIZE = 29  # 28-byte name + newline; even with the leading '/', names fit Darwin's limit.


def _default_root() -> Path:
    # Only the storage location differs; allocation, journaling and reclamation are identical.
    # Unlike /tmp or Caches, these directories are not periodically purged while an app runs.
    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support" / "starfish"
    else:
        root = Path.home() / ".local" / "state" / "starfish"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return root


def _compact_name(name: str) -> str:
    return "sfm_" + hashlib.sha256(name.encode()).hexdigest()[:24]


def session_queue_name(name: str) -> str:
    """Assign compact session-owned names without changing logical node/port identities."""
    manifest = _MANIFEST.get()
    if manifest is not None:
        name = _compact_name(f"{os.getuid()}_{manifest.stem}_{name}")
        _QUEUE_MANIFESTS[name] = manifest
    return name


def queue_frame_name(queue_name: str, index: int) -> str:
    """Both producers and consumers derive bounded-length names, even for large frame indices."""
    name = f"{queue_name}_{index}"
    return _compact_name(name) if _SEGMENT_NAME.fullmatch(queue_name) else name


def register_segment(queue_name: str, name: str) -> None:
    """Journal a segment BEFORE shm_open; SIGKILL must never leave an unrecorded object."""
    manifest = _QUEUE_MANIFESTS.get(queue_name)
    if manifest is None:
        return
    if not _SEGMENT_NAME.fullmatch(name):
        raise ValueError(f"Invalid session shared-memory name: {name!r}")
    fd = os.open(manifest, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        # A killed writer may leave an incomplete record, but cannot have created its segment yet.
        size = os.fstat(fd).st_size
        if size % _RECORD_SIZE:
            os.ftruncate(fd, size - size % _RECORD_SIZE)
        if os.write(fd, (name + "\n").encode("ascii")) != _RECORD_SIZE:
            raise OSError("Incomplete shared-memory ownership record")
    finally:
        os.close(fd)


def _reclaim_manifest(manifest: Path) -> tuple[int, bool]:
    removed = 0
    complete = True
    with manifest.open("rb") as records:
        for record in records:
            if len(record) < _RECORD_SIZE and not record.endswith(b"\n"):
                continue  # Incomplete final write: no segment could have been created for it.
            name = record[:-1].decode("ascii", errors="replace")
            if len(record) != _RECORD_SIZE or not record.endswith(b"\n") or not _SEGMENT_NAME.fullmatch(name):
                raise OSError(f"Invalid shared-memory ownership record in {manifest}")
            try:
                # Unlink by name, without attaching/mapping (also handles zero-sized crashed allocations).
                _posixshmem.shm_unlink("/" + name)
                removed += 1
            except FileNotFoundError:
                pass
            except OSError:
                complete = False
                _LOG.warning("Could not reclaim session segment %s", name, exc_info=True)
    return removed, complete


@contextmanager
def _registry(registry_root: Path) -> Iterator[Path]:
    directory = registry_root / f"starfish-sessions-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise PermissionError(f"Unsafe shared-memory session registry: {directory}")
    fd = os.open(directory / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield directory
    finally:
        os.close(fd)


def _reclaim(lease: Path) -> int:
    """Called under the registry lock; skip any lease still held by a runner or worker."""
    try:
        fd = os.open(lease, os.O_RDWR | os.O_NOFOLLOW)
    except FileNotFoundError:
        return 0  # Another scavenger may have reclaimed it after the owner closed its descriptor.
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        manifest = lease.with_suffix(".segments")
        try:
            removed, complete = _reclaim_manifest(manifest)
        except FileNotFoundError:
            # The owner may have died before publishing its manifest; no allocation could follow.
            lease.unlink()
            return 0
        if complete:
            manifest.unlink()
            lease.unlink()
        return removed
    finally:
        os.close(fd)


def cleanup_abandoned_sessions(registry_root: Path | None = None) -> int:
    """Reclaim owned index/frame segments only; never guess ownership of legacy segments."""
    registry_root = _default_root() if registry_root is None else registry_root
    removed = 0
    with _registry(registry_root) as directory:
        for lease in directory.iterdir():
            if re.fullmatch(r"[0-9a-f]{32}\.lock", lease.name):
                try:
                    removed += _reclaim(lease)
                except OSError:
                    _LOG.warning("Could not inspect session lease %s", lease, exc_info=True)
    if removed:
        _LOG.info("Reclaimed %d abandoned shared-memory segment(s)", removed)
    return removed


@contextmanager
def shared_memory_session(registry_root: Path | None = None) -> Iterator[None]:
    """Own all OutputStage queues created during graph materialization and worker execution."""
    registry_root = _default_root() if registry_root is None else registry_root
    session_id = uuid.uuid4().hex
    # Serialize publication with scavengers so an unlocked, not-yet-started session is never reaped.
    with _registry(registry_root) as directory:
        lease = directory / f"{session_id}.lock"
        fd = os.open(lease, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            manifest = lease.with_suffix(".segments")
            manifest.touch(mode=0o600, exist_ok=False)
        except BaseException:
            os.close(fd)
            raise
    manifest_token = _MANIFEST.set(manifest)
    try:
        yield
    finally:
        _MANIFEST.reset(manifest_token)
        for name in [name for name, owner in _QUEUE_MANIFESTS.items() if owner == manifest]:
            del _QUEUE_MANIFESTS[name]
        # Never LOCK_UN here: forked workers share this open file description and may still be live.
        os.close(fd)
        try:
            with _registry(registry_root):
                _reclaim(lease)
        except OSError:
            # Retain ownership metadata for a later retry; never hide the original graph failure.
            _LOG.warning("Could not reclaim session %s on exit", session_id, exc_info=True)
