# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import pickle
import socket
import struct
import threading
import unittest
from typing import Any
from unittest.mock import Mock, patch

import numpy as np

from judo.unlocked.nodes.network import Receiver, Sender
from judo.unlocked.shmque import Memory


class TestNetwork(unittest.TestCase):
    """Exercise framing and lifecycle on real loopback sockets, without fixed ports."""

    def receiver(self, **kwargs: Any) -> Receiver:
        receiver = Receiver("127.0.0.1", 0, timeout=0.03, **kwargs)
        receiver.start()
        self.addCleanup(receiver.stop)
        return receiver

    def peer(self, receiver: Receiver) -> socket.socket:
        assert receiver._server_socket is not None
        peer = socket.create_connection(receiver._server_socket.getsockname(), timeout=1)
        self.addCleanup(peer.close)
        return peer

    def sender(self, receiver: Receiver, **kwargs: Any) -> Sender:
        assert receiver._server_socket is not None
        sender = Sender(*receiver._server_socket.getsockname(), timeout=0.03, **kwargs)
        sender.start()
        self.addCleanup(sender.stop)
        return sender

    @staticmethod
    def frame(value: Any) -> bytes:
        memory: Memory[Any] = Memory(value)
        body = bytearray(len(memory))
        memory.encode(body)
        return struct.pack("!Q", len(body)) + body

    @staticmethod
    def read_exact(peer: socket.socket, size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            chunk = peer.recv(size - len(data))
            if not chunk:
                raise AssertionError("Unexpected EOF")
            data.extend(chunk)
        return bytes(data)

    def test_fragmented_header_and_body_survive_timeouts(self) -> None:
        receiver = self.receiver()
        peer = self.peer(receiver)
        value = {"command": "move", "values": [1, True, None, 2.5]}
        frame = self.frame(value)
        start = 0
        for end in (1, 4, 7, 8, 10, len(frame) - 1):
            peer.sendall(frame[start:end])
            self.assertIsNone(receiver())
            start = end
        peer.sendall(frame[start:])
        self.assertEqual(receiver(), value)
        # A subsequent frame must not contain any bytes from the old one.
        peer.sendall(self.frame(["next"]))
        self.assertEqual(receiver(), ["next"])

    def test_coalesced_frames(self) -> None:
        receiver = self.receiver()
        peer = self.peer(receiver)
        peer.sendall(self.frame({"first": 1}) + self.frame({"second": 2}))
        self.assertEqual(receiver(), {"first": 1})
        self.assertEqual(receiver(), {"second": 2})

    def test_receive_deadline_retains_progress_even_with_continuous_data(self) -> None:
        receiver = self.receiver()
        peer = self.peer(receiver)
        self.assertIsNone(receiver())  # Accept the idle peer first.
        connection = receiver._connection
        assert connection is not None
        wrapped = Mock(wraps=connection)
        receiver._connection = wrapped
        value = {"command": "move"}
        peer.sendall(self.frame(value))
        # Force short successful reads; the deadline must still end this call.
        wrapped.recv.side_effect = lambda size: connection.recv(min(size, 4))
        with patch("judo.unlocked.nodes.network.time.monotonic", side_effect=[10, 10, 10.01, 10.02, 10.04]):
            self.assertIsNone(receiver())
        self.assertEqual(len(receiver._buffer), 4)
        self.assertIs(receiver._connection, wrapped)
        self.assertEqual(receiver(), value)

    def test_idle_accept_and_connected_receive_stop_without_hanging(self) -> None:
        for connected in (False, True):
            with self.subTest(connected=connected):
                receiver = self.receiver()
                if connected:
                    peer = self.peer(receiver)
                    self.assertIsNone(receiver())
                    assert receiver._connection is not None
                    timeout = receiver._connection.gettimeout()
                    assert timeout is not None
                    self.assertGreater(timeout, 0)
                finished = threading.Event()
                entered = threading.Event()
                errors: list[Any] = []

                def receive(
                    receiver: Receiver = receiver,
                    entered: Any = entered,
                    errors: list[Any] = errors,
                    finished: Any = finished,
                ) -> None:
                    entered.set()
                    try:
                        receiver()
                    except Exception as error:
                        errors.append(error)
                    finally:
                        finished.set()

                worker = threading.Thread(target=receive, daemon=True)
                worker.start()
                self.assertTrue(entered.wait(1))
                receiver.stop()
                self.assertTrue(finished.wait(1), "stop left a blocked receiver")
                worker.join(1)
                self.assertEqual(errors, [])
                receiver.stop()
                if connected:
                    peer.close()

    def test_eof_discards_partial_frame_and_accepts_reconnect(self) -> None:
        for cutoff in (3, 10):
            with self.subTest(cutoff=cutoff):
                receiver = self.receiver()
                peer = self.peer(receiver)
                peer.sendall(self.frame({"incomplete": True})[:cutoff])
                peer.shutdown(socket.SHUT_WR)
                self.assertIsNone(receiver())
                self.assertIsNone(receiver._connection)
                self.assertEqual(len(receiver._buffer), 0)
                replacement = self.peer(receiver)
                replacement.sendall(self.frame({"complete": True}))
                self.assertEqual(receiver(), {"complete": True})

    def test_stop_between_accept_and_connection_snapshot(self) -> None:
        receiver = self.receiver()
        self.peer(receiver)
        # This log occurs after accept's server identity check but before the snapshot.
        with patch("judo.unlocked.nodes.network.logging.info", side_effect=lambda *args: receiver.stop()):
            self.assertIsNone(receiver())
        self.assertIsNone(receiver._connection)

    def test_request_reply_roundtrip(self) -> None:
        receiver = self.receiver()
        sender = self.sender(receiver)
        errors: list[Exception] = []

        def serve() -> None:
            try:
                self.assertEqual(receiver(), {"request": 1})
                receiver.reply({"result": 2})
            except Exception as error:
                errors.append(error)

        worker = threading.Thread(target=serve, daemon=True)
        worker.start()
        self.assertEqual(sender.request({"request": 1}, timeout=1), {"result": 2})
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])

    def test_request_timeout_discards_connection_and_late_reply(self) -> None:
        receiver = self.receiver()
        sender = self.sender(receiver)
        with self.assertRaises(TimeoutError):
            sender.request({"request": 1}, timeout=0.02)
        self.assertIsNone(sender._client_socket)
        self.assertEqual(receiver(), {"request": 1})
        self.assertIsNone(receiver())  # EOF; the timed-out stream cannot be reused.
        sender({"next": 2})
        self.assertEqual(receiver(), {"next": 2})

    def test_strict_send_propagates_delivery_failures(self) -> None:
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            sender = Sender(*reserved.getsockname())
            self.addCleanup(sender.stop)
            with self.assertRaisesRegex(ConnectionError, "stopped"):
                sender.send("command", raise_on_error=True)
            sender.start()
            with self.assertRaises(OSError):
                sender.send("command", raise_on_error=True)
            self.assertIsNone(sender._client_socket)

    def test_request_partial_send_failure_is_not_retried(self) -> None:
        receiver = self.receiver()
        sender = self.sender(receiver)
        sender("first")
        self.assertEqual(receiver(), "first")
        original = sender._client_socket
        assert original is not None
        failing = Mock(wraps=original)

        def partial_send(data: bytes) -> None:
            original.sendall(data[:10])
            raise TimeoutError("partial send")

        failing.sendall.side_effect = partial_send
        sender._client_socket = failing
        with self.assertRaisesRegex(TimeoutError, "partial send"):
            sender.request("command", timeout=1)
        failing.sendall.assert_called_once()
        self.assertIsNone(sender._client_socket)
        self.assertIsNone(receiver())

    def test_oversized_header_is_rejected_before_reading_body(self) -> None:
        receiver = self.receiver(max_frame_bytes=128)
        peer = self.peer(receiver)
        peer.sendall(struct.pack("!Q", 2**64 - 1))
        with self.assertRaisesRegex(ValueError, "frame"):
            receiver()
        self.assertIsNone(receiver._connection)
        self.assertEqual(len(receiver._buffer), 0)
        self.assertEqual(peer.recv(1), b"")
        replacement = self.peer(receiver)
        replacement.sendall(self.frame({"ok": True}))
        self.assertEqual(receiver(), {"ok": True})

    def test_invalid_payload_is_not_silently_ignored(self) -> None:
        for body in (b"bad", b"", struct.pack("NN", 1, 3) + b"bad"):
            with self.subTest(body=body):
                receiver = self.receiver()
                peer = self.peer(receiver)
                peer.sendall(struct.pack("!Q", len(body)) + body)
                with self.assertRaises((ValueError, struct.error, pickle.UnpicklingError)):
                    receiver()
                self.assertIsNone(receiver._connection)

    def test_default_memory_roundtrip(self) -> None:
        receiver = self.receiver()
        sender = self.sender(receiver)
        for value in (
            {"robot": "spot", "values": [1, 2.5, True, None, "é"]},
            ["stop", False],
            "hello",
            42,
            2.5,
            True,
            None,
        ):
            sender(value)
            self.assertEqual(receiver(), value)

    def test_nested_ndarray_roundtrip_and_buffer_ownership(self) -> None:
        receiver = self.receiver()
        sender = self.sender(receiver)
        value = {
            "telemetry": {
                "joints": np.arange(19, dtype=np.float32),
                "pose": np.eye(4),
                "bodies": {107: np.arange(6).reshape(2, 3)},
            }
        }
        sender(value)
        result = receiver()
        sender({"next": np.ones(19)})
        self.assertIsNotNone(receiver())
        for name, expected in value["telemetry"].items():
            if isinstance(expected, dict):
                np.testing.assert_array_equal(result["telemetry"][name][107], expected[107])
            else:
                np.testing.assert_array_equal(result["telemetry"][name], expected)
                self.assertEqual(result["telemetry"][name].dtype, expected.dtype)

    def test_sender_wire_header_is_network_order(self) -> None:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(1)
            sender = Sender(*listener.getsockname())
            self.addCleanup(sender.stop)
            sender.start()
            sender({"command": "stop"})
            peer, _ = listener.accept()
            with peer:
                peer.settimeout(1)
                size = struct.unpack("!Q", self.read_exact(peer, 8))[0]
                self.assertLess(size, 100)
                self.assertEqual(Memory.decode(bytearray(self.read_exact(peer, size))), {"command": "stop"})

    def test_failed_connect_recreates_socket(self) -> None:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))  # Reserve a port without listening yet.
            listener.settimeout(1)
            sender = Sender(*listener.getsockname(), timeout=0.03)
            self.addCleanup(sender.stop)
            sender.start()
            failed_socket = sender._client_socket
            assert failed_socket is not None
            sender({"lost": True})
            self.assertEqual(failed_socket.fileno(), -1)
            self.assertFalse(sender._client_connected)
            listener.listen(1)
            sender({"retry": True})
            peer, _ = listener.accept()
            with peer:
                peer.settimeout(1)
                size = struct.unpack("!Q", self.read_exact(peer, 8))[0]
                self.assertEqual(Memory.decode(bytearray(self.read_exact(peer, size))), {"retry": True})

    def test_connect_uses_generous_budget_then_drops_to_short_io_timeout(self) -> None:
        # The handshake gets the roomy connect_timeout (a 2 ms per-frame budget can't complete a
        # connect() over WiFi), then the connected socket drops to the short per-frame I/O timeout.
        receiver = self.receiver()
        sender = self.sender(receiver, connect_timeout=0.5)
        assert sender._client_socket is not None
        self.assertEqual(sender._client_socket.gettimeout(), 0.5)  # fresh socket: connect budget
        sender({"hello": 1})
        self.assertEqual(receiver(), {"hello": 1})
        self.assertTrue(sender._client_connected)
        assert sender._client_socket is not None
        self.assertEqual(sender._client_socket.gettimeout(), 0.03)  # connected: short I/O budget

    def test_partial_send_failure_closes_socket_and_reconnects(self) -> None:
        receiver = self.receiver()
        sender = self.sender(receiver)
        sender({"first": 1})
        self.assertEqual(receiver(), {"first": 1})
        original = sender._client_socket
        assert original is not None
        failing = Mock(wraps=original)

        def partial_send(data: Any) -> None:
            assert original is not None
            original.sendall(data[:10])
            raise socket.timeout("injected partial send timeout")

        failing.sendall.side_effect = partial_send
        sender._client_socket = failing
        sender({"incomplete": True})
        self.assertEqual(original.fileno(), -1)
        self.assertFalse(sender._client_connected)
        self.assertIsNone(receiver())  # Discard the incomplete frame at EOF.
        sender({"reconnected": True})
        self.assertEqual(receiver(), {"reconnected": True})

    def test_sender_size_limit_and_encoding_failure_leave_stream_intact(self) -> None:
        receiver = self.receiver()
        sender = self.sender(receiver, max_frame_bytes=128)
        with self.assertRaisesRegex(ValueError, "frame"):
            sender("x" * 1000)
        sender("ok")
        self.assertEqual(receiver(), "ok")
        with self.assertRaises((AttributeError, pickle.PicklingError)):
            sender(lambda: None)
        sender({"ok": True})
        self.assertEqual(receiver(), {"ok": True})

    def test_frame_at_exact_limit(self) -> None:
        size = len(Memory("ok"))
        receiver = self.receiver(max_frame_bytes=size)
        sender = self.sender(receiver, max_frame_bytes=size)
        sender("ok")
        self.assertEqual(receiver(), "ok")

    def test_lifecycle_is_idempotent_and_restartable(self) -> None:
        Sender("127.0.0.1", 0).stop()
        receiver = Receiver("127.0.0.1", 0, timeout=0.03)
        receiver.stop()
        receiver.start()
        self.addCleanup(receiver.stop)
        listener = receiver._server_socket
        assert listener is not None
        address = listener.getsockname()
        self.assertNotEqual(listener.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR), 0)
        receiver.start()
        self.assertIs(receiver._server_socket, listener)
        sender = self.sender(receiver)
        sender.start()
        sender({"first": True})
        self.assertEqual(receiver(), {"first": True})
        connected = sender._client_socket
        sender.start()
        self.assertIs(sender._client_socket, connected)
        sender.stop()
        sender.stop()
        receiver.stop()
        receiver.stop()
        # Rebind the same port immediately, not another ephemeral port.
        receiver._port = address[1]
        receiver.start()
        sender.start()
        sender({"restarted": True})
        self.assertEqual(receiver(), {"restarted": True})
        sender.stop()
        sender({"stopped": True})
        self.assertIsNone(sender._client_socket)
        receiver.stop()
        self.assertIsNone(receiver())

    def test_invalid_configuration(self) -> None:
        for cls in (Sender, Receiver):
            for options in (
                {"max_frame_bytes": 0},
                {"max_frame_bytes": -1},
                {"max_frame_bytes": 1.5},
                {"timeout": 0},
                {"timeout": -1},
                {"timeout": float("inf")},
                {"timeout": float("nan")},
            ):
                with self.subTest(cls=cls, options=options), self.assertRaises(ValueError):
                    cls("127.0.0.1", 0, **options)


if __name__ == "__main__":
    unittest.main()
