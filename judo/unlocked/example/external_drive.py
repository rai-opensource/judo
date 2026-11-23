# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Externally driven nodes: ``Node.callback`` (push in) and ``Node.generator`` (pull out).

Most nodes are driven BY the scheduler: it gives each one a thread that loops
``wait -> exec -> wait``. Sometimes the loop already exists somewhere else -- a GUI toolkit's
event thread, or a streaming client that wants a Python generator -- and adding a scheduler
thread on top just means an extra hand-off, an extra wakeup and a mailbox to keep in sync.

``node_type=NodeType.CALLBACK`` / ``NodeType.GENERATOR`` says "this node is driven from outside". The
scheduler still owns its
lifecycle (stages opened and closed, ``warmup`` published, ``close`` run in order), it simply
spawns no worker for it. Two ways to drive one:

* :meth:`Node.callback` -- an external producer PUSHES. Call it with the target's arguments and
  the result goes straight into the output queue, on the caller's thread. This example's
  ``_button_thread`` stands in for a GUI click handler.
* :meth:`Node.generator` -- an external consumer PULLS. Iterating it blocks on the input queue,
  runs the target, and yields the result, so the consumer's own loop paces the node. This
  example's ``_stream_thread`` stands in for a client that must be handed an iterator.

Between the two sits ``brightness``, an ordinary UNLOCKED node, to show that externally driven
and scheduler-driven nodes compose in one graph:

    button (callback)  ->  brightness (unlocked)  ->  stream (generator)
      ^ pushed by an                                     ^ pulled by an
        external thread                                    external thread

Run: ``bazel run //judo/unlocked/example:external_drive`` (Ctrl-C to stop).
"""

import logging
import signal
import threading
import time

from judo.unlocked.logging_util import setup_logger
from judo.unlocked import Node, NodeType
from judo.unlocked.policy import Lossless
from judo.unlocked.schedule import StopAll, Threaded

setup_logger(log_level=logging.INFO)

_STOP = threading.Event()


def button_press(level: int) -> dict:
    """Target of the CALLBACK node: turn one external event into a published message.

    Runs on whichever thread calls ``Node.callback`` -- here the button thread, in a real app a
    viser / Qt handler. Keep it cheap and non-blocking: it is on the caller's critical path.
    """
    return {"level": level}


def brightness(command: dict) -> dict:
    """Ordinary scheduled node: the scheduler gives THIS one a worker thread."""
    level = int(command["level"])
    return {"level": level, "percent": min(100, level * 10)}


def _button_thread(node: Node) -> None:
    """Stand-in for a GUI event loop: PUSH into the graph via ``callback``."""
    level = 0
    while not _STOP.is_set():
        level += 1
        logging.info("[button]  press -> level %d", level)
        # Publishes on THIS thread: target runs, result lands in the shm queue. No scheduler
        # thread is involved, and nothing is buffered waiting for one to wake up.
        node.callback(level)
        _STOP.wait(1.0)


def _stream_thread(node: Node) -> None:
    """Stand-in for a streaming client: PULL out of the graph via ``generator``.

    A client that wants ``send(iterator)`` can be handed ``node.generator()`` directly -- each
    ``next()`` blocks until the upstream node publishes, so the producer paces the stream.
    """
    for message in node.generator():
        logging.info("[stream]  <- %s", dict(message))
    # The loop ends when the node stops being live AND its park is released, which is exactly
    # what scheduler shutdown does (close, then wake). See ``main``'s teardown below.
    logging.info("[stream]  generator ended; consumer exiting")


def main() -> None:
    """Wire one callback-driven node, one unlocked node and one generator-driven node."""
    # A non-default ``node_type`` says the node is driven from outside: the scheduler manages both
    # but gives them no worker thread. CALLBACK is pushed into; GENERATOR is pulled from.
    button = Node("button", button_press, node_type=NodeType.CALLBACK)
    stream = Node("stream", brightness, node_type=NodeType.GENERATOR)
    # A normal node, for contrast: this one DOES get a scheduler thread.
    lamp = Node("lamp", brightness)

    lamp.input_stage.connect((button.output_stage[0], Lossless()))
    stream.input_stage.connect((lamp.output_stage[0], Lossless()))

    schedule = Threaded("external_drive", (button, lamp, stream))
    # Run the scheduler in THIS process so the external threads below share its nodes.
    # ``Threaded.start`` would fork, and the forked child's nodes would be copies.
    spin = threading.Thread(target=schedule.spin, name="scheduler", daemon=True)
    spin.start()
    # ``spin`` assigns ``node_threads`` once every stage is open -- wait for that before driving.
    while getattr(schedule, "node_threads", None) is None:
        time.sleep(0.01)
    logging.info(
        "scheduler worker threads: %s (button/stream are externally driven, so absent)",
        sorted(t.name for t in schedule.node_threads),
    )

    threads = [
        threading.Thread(target=_button_thread, args=(button,), name="button_ext", daemon=True),
        threading.Thread(target=_stream_thread, args=(stream,), name="stream_ext", daemon=True),
    ]
    for thread in threads:
        thread.start()

    def _shutdown(*_: object) -> None:
        """Stop the external producer first, then release the parked consumer."""
        _STOP.set()
        # A generator consumer parks in an UNTIMED wait, so marking the node not-live is not
        # enough on its own -- wake() releases the park so the iterator can end.
        stream.close()
        stream.wake()
        StopAll((schedule,))(signal.SIGINT, None)

    signal.signal(signal.SIGINT, _shutdown)
    spin.join()
    _STOP.set()
    for thread in threads:
        thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
