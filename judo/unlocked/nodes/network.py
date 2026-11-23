# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Framed Memory/pickle TCP for trusted hosts with matching Python/native architectures.

This is not authenticated or safe for untrusted input. Never expose these listeners
to untrusted networks; SSH process supervision does not authenticate this TCP path.
"""

import logging
import math
import socket
import struct
import time
from typing import Any, cast

from judo.unlocked.const import const_cast
from judo.unlocked.node import Node
from judo.unlocked.shmque import Memory

MAX_FRAME_BYTES = 64 * 1024 * 1024
_HEADER = struct.Struct("!Q")
_SOCKET_TIMEOUT = 0.1
# The TCP handshake gets its OWN, generous budget: the per-frame ``timeout`` is deliberately tiny
# (poll-friendly, non-blocking I/O for realtime threads), but a 2 ms budget is far too short to
# complete a connect() over a WiFi LAN, so a Sender would churn connect->timeout->retry forever and
# never establish the channel (e.g. the desktop's COMMAND/GOAL senders to the robots). A connect
# blocks a FRESH socket on a dedicated node thread (never the already-connected steady-state send
# path), and ``socket.connect`` releases the GIL while it waits, so a roomy budget here is safe.
_CONNECT_TIMEOUT = 1.0


def _validate_options(max_frame_bytes: int, timeout: float) -> None:
    if type(max_frame_bytes) is not int or not 0 < max_frame_bytes < 2**64:
        raise ValueError("max_frame_bytes must be a positive uint64")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")


def _close_socket(sock: socket.socket | None) -> None:
    if sock is not None:
        try:
            # Wake an outstanding recv/accept, including when stop runs on another thread.
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()


def make_receiver_node(
    name: str,
    host: str,
    port: int,
    *,
    max_frame_bytes: int = MAX_FRAME_BYTES,
    timeout: float = _SOCKET_TIMEOUT,
) -> Node:
    """Make a receiver node."""
    receiver = Receiver(host, port, max_frame_bytes=max_frame_bytes, timeout=timeout)
    return Node(name=name, target=receiver, pre_open=receiver.start, post_close=receiver.stop)


def make_sender_node(
    name: str,
    host: str,
    port: int,
    *,
    max_frame_bytes: int = MAX_FRAME_BYTES,
    timeout: float = _SOCKET_TIMEOUT,
) -> Node:
    """Make a sender node."""
    sender = Sender(host, port, max_frame_bytes=max_frame_bytes, timeout=timeout)
    return Node(name=name, target=sender, pre_open=sender.start, post_close=sender.stop)


class Receiver:
    """Receive one length-prefixed TCP message per call, or None on timeout/EOF.

    Both peers use Memory serialization (pickle; trusted LAN and matching native layout only).
    The wire header is an unsigned 8-byte network-order payload length, not the
    legacy native-size header. Partial frames survive timeouts, but are discarded
    on disconnect. Oversized/invalid frames close the connection and raise.
    Calls have a bounded I/O budget; call again to resume an incomplete frame.
    Use a single receiving caller; stop may also be called from another thread.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        max_frame_bytes: int = MAX_FRAME_BYTES,
        timeout: float = _SOCKET_TIMEOUT,
    ) -> None:
        _validate_options(max_frame_bytes, timeout)
        self._host = host
        self._port = port
        self._max_frame_bytes = max_frame_bytes
        self._timeout = timeout
        self._server_socket: socket.socket | None = None
        self._connection: socket.socket | None = None
        self._buffer = bytearray()
        self._frame_size: int | None = None

    def start(self) -> None:
        """Open server socket and listen for connection."""
        if self._server_socket is not None:
            return
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.settimeout(self._timeout)
            server.bind((self._host, self._port))
            server.listen(1)
        except OSError:
            server.close()
            raise
        self._server_socket = server

    def _disconnect(self) -> None:
        connection, self._connection = self._connection, None
        self._buffer = bytearray()
        self._frame_size = None
        _close_socket(connection)

    def stop(self) -> None:
        """Close all sockets and discard partial data; safe before start or repeatedly."""
        server, self._server_socket = self._server_socket, None
        self._disconnect()
        _close_socket(server)

    def _receive_frame(self, connection: socket.socket, deadline: float) -> bytearray | None:
        while connection is self._connection:
            size = _HEADER.size if self._frame_size is None else self._frame_size
            if len(self._buffer) == size:
                if self._frame_size is None:
                    self._frame_size = _HEADER.unpack(self._buffer)[0]
                    self._buffer = bytearray()
                    if self._frame_size > self._max_frame_bytes:
                        raise ValueError(f"TCP frame exceeds {self._max_frame_bytes} bytes")
                    continue
                buffer, self._buffer = self._buffer, bytearray()
                self._frame_size = None
                return buffer
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            connection.settimeout(remaining)
            # Never allocate from an unchecked header, nor over-read the next frame.
            packet = connection.recv(min(size - len(self._buffer), 64 * 1024))
            if not packet:
                raise OSError("EOF")
            if connection is not self._connection:
                return None
            self._buffer.extend(packet)
        return None

    def __call__(self) -> Any:
        """Receive data, decode and publish."""
        server = self._server_socket
        if server is None:
            return None
        deadline = time.monotonic() + self._timeout
        if self._connection is None:
            try:
                connection, (host, port) = server.accept()
                connection.settimeout(self._timeout)
                self._connection = connection
                if self._server_socket is not server:
                    self._disconnect()
                    return None
                logging.info("Receiver is connected to host %s on port %d", host, port)
            except OSError:
                return None
        # stop() can clear this between accept's check and the snapshot, on another thread.
        captured_connection = cast(socket.socket | None, self._connection)
        if captured_connection is None:
            return None
        try:
            buffer = self._receive_frame(captured_connection, deadline)
        except socket.timeout:
            # Keep both header/body progress and the connection for the next call.
            return None
        except OSError:
            self._disconnect()
            logging.info("Receiver has disconnected")
            return None
        except Exception:
            self._disconnect()
            raise
        if buffer is None:
            return None
        try:
            # Memory payloads retain their native layout and trusted-peer assumptions.
            return Memory.decode(buffer)
        except Exception:
            self._disconnect()
            raise

    def reply(self, message: Any) -> None:
        """Reply on the current connection; a failed reply invalidates the stream."""
        connection = self._connection
        if connection is None:
            raise ConnectionError("Receiver has no connection for reply")
        try:
            memory: Memory = Memory(message)
            if len(memory) > self._max_frame_bytes:
                raise ValueError(f"TCP frame exceeds {self._max_frame_bytes} bytes")
            buffer = bytearray(len(memory))
            memory.encode(buffer)
            connection.settimeout(self._timeout)
            connection.sendall(_HEADER.pack(len(buffer)) + buffer)
        except Exception:
            self._disconnect()
            raise


class Sender:
    """Send bounded !Q-prefixed Memory payloads (pickle; trusted LAN only).

    Connection/send failures discard the socket, not retry the message: a partial
    send has ambiguous delivery. The next call reconnects using a fresh socket.
    Invalid/oversized local payloads raise without sending any frame bytes.
    Lifecycle and sending calls should be serialized by the caller.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        max_frame_bytes: int = MAX_FRAME_BYTES,
        timeout: float = _SOCKET_TIMEOUT,
        connect_timeout: float = _CONNECT_TIMEOUT,
    ) -> None:
        _validate_options(max_frame_bytes, timeout)
        if not math.isfinite(connect_timeout) or connect_timeout <= 0:
            raise ValueError("connect_timeout must be positive and finite")
        self._host = host
        self._port = port
        self._max_frame_bytes = max_frame_bytes
        self._timeout = timeout
        self._connect_timeout = connect_timeout
        self._client_socket: socket.socket | None = None
        self._client_connected = False
        self._started = False

    def _open_socket(self) -> None:
        self._client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Start on the generous CONNECT budget; __call__ switches to the short per-frame budget once
        # the handshake completes so steady-state sends stay non-blocking for realtime threads.
        self._client_socket.settimeout(self._connect_timeout)

    def _disconnect(self) -> None:
        client, self._client_socket = self._client_socket, None
        self._client_connected = False
        _close_socket(client)

    def start(self) -> None:
        """Open connection."""
        if self._started:
            return
        self._open_socket()
        self._started = True

    def stop(self) -> None:
        """Close connection."""
        self._started = False
        self._disconnect()

    def _encode(self, msg: Any) -> bytearray:
        msg = const_cast(msg)
        mem: Memory = Memory(msg)
        size = len(mem)
        if size > self._max_frame_bytes:
            raise ValueError(f"TCP frame exceeds {self._max_frame_bytes} bytes")
        buffer = bytearray(size)
        mem.encode(buffer)
        return buffer

    def __call__(self, msg: Any) -> None:
        """Stream data over TCP connection."""
        self.send(msg)

    def send(self, msg: Any, *, raise_on_error: bool = False) -> None:
        """Send once, optionally propagating delivery failures (never retry a partial send)."""
        if not self._started:
            if raise_on_error:
                raise ConnectionError("Sender is stopped")
            return
        buffer = self._encode(msg)
        try:
            if self._client_socket is None:
                self._open_socket()
            client = self._client_socket
            assert client is not None
            if not self._client_connected:
                client.connect((self._host, self._port))
                client.settimeout(self._timeout)  # connected: drop to the short per-frame I/O budget
                self._client_connected = True
                logging.info("Sender is connected to host %s on port %d", self._host, self._port)
            client.sendall(_HEADER.pack(len(buffer)) + buffer)
        except OSError:
            self._disconnect()
            if raise_on_error:
                raise

    def request(self, message: Any, *, timeout: float) -> Any:
        """Send a request and await one framed reply; serialize calls with sending/stopping.

        A timeout discards the connection so a late reply cannot satisfy a later request.
        Sending is never retried: a failed response may follow successful remote execution.
        """
        _validate_options(self._max_frame_bytes, timeout)
        self.send(message, raise_on_error=True)
        client = self._client_socket
        assert client is not None
        reader = Receiver("", 0, max_frame_bytes=self._max_frame_bytes, timeout=timeout)
        # Share this module's incremental frame parser, but not the listening socket lifecycle.
        reader._connection = client  # noqa: SLF001
        try:
            buffer = reader._receive_frame(client, time.monotonic() + timeout)  # noqa: SLF001
            if buffer is None:
                raise TimeoutError("Timed out waiting for network response")
            return Memory.decode(buffer)
        except Exception:
            self._disconnect()
            raise
        finally:
            if self._client_socket is client:
                client.settimeout(self._timeout)
