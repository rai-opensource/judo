# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import random
import unittest
from dataclasses import dataclass
from enum import Enum, auto
from threading import Thread
from time import sleep
from typing import Any, Callable

from parameterized import parameterized_class

from judo.unlocked import Queue, QueueView


@dataclass
class MyInt:
    value: int


type_params = [int, MyInt]


def gen_int(_: Any, value: int) -> int:
    return value


def gen_my_int(_: Any, value: int) -> MyInt:
    return MyInt(value)


@parameterized_class(("T"), [(gen_int,), (gen_my_int,)])
class TestQueue(unittest.TestCase):
    """Test queue."""

    T: Callable

    def test_queue(self) -> None:
        q: Queue = Queue()
        q.push(self.T(8))
        q.push(self.T(7))
        self.assertEqual(q[0], self.T(8))
        self.assertEqual(q[1], self.T(7))
        self.assertEqual(q.top(), self.T(8))
        self.assertEqual(q.pop(), self.T(8))
        self.assertEqual(q[1], self.T(7))
        with self.assertRaises(IndexError):
            q[0]
        self.assertEqual(q.pop(), self.T(7))
        with self.assertRaises(IndexError):
            q[0]
        with self.assertRaises(IndexError):
            q[1]
        with self.assertRaises(IndexError):
            q.pop()
        q.push(self.T(3))
        with self.assertRaises(IndexError):
            q[0]
        with self.assertRaises(IndexError):
            q[1]
        self.assertEqual(q[2], self.T(3))

    def test_queue_view(self) -> None:
        q: Queue = Queue()
        qw = q.make_view()

        q.push(self.T(8))
        q.push(self.T(7))
        with self.assertRaises(IndexError):
            qw.last()
        with self.assertRaises(IndexError):
            qw.top()
        with self.assertRaises(IndexError):
            qw.pop()
        qw.flush()
        q.flush()
        self.assertEqual(len(q), 2)
        self.assertEqual(qw.top(), self.T(8))
        self.assertEqual(qw.last(), self.T(7))
        self.assertEqual(qw.pop(), self.T(8))
        self.assertEqual(qw.top(), self.T(7))
        self.assertEqual(qw.last(), self.T(7))
        q.push(self.T(3))
        self.assertEqual(len(q), 3)
        self.assertEqual(qw.top(), self.T(7))
        self.assertEqual(qw.last(), self.T(7))
        q.flush()
        self.assertEqual(len(q), 2)
        qw.flush(flush_queue=True)
        self.assertEqual(len(q), 1)
        self.assertEqual(qw.pop(), self.T(3))
        q.flush()
        self.assertEqual(len(q), 0)

    def test_queue_stress(self) -> None:
        q: Queue = Queue()

        random.seed(83285020238474093)

        class Act(Enum):
            Push = auto()
            Pop = auto()
            Item = auto()

        test_num = 100
        act_num = 1000

        for _ in range(test_num):
            r_seq = [random.randint(0, 1_000_000_000) for _ in range(act_num)]
            push_pos = 0
            pop_pos = 0
            q_len = 0
            q.clear()
            for act in random.choices(list(Act.__members__.values()), k=act_num):
                match act:
                    case Act.Push:
                        q.push(self.T(r_seq[push_pos]))
                        push_pos += 1
                        q_len += 1
                    case Act.Pop:
                        if q_len > 0:
                            self.assertEqual(q.pop(), self.T(r_seq[pop_pos]))
                            pop_pos += 1
                            q_len -= 1
                        else:
                            with self.assertRaises(IndexError):
                                q.pop()
                    case Act.Item:
                        i = random.randint(0, act_num)
                        if q.begin <= i and i < q.end:
                            self.assertEqual(q[i], self.T(r_seq[i]))
                        else:
                            with self.assertRaises(IndexError):
                                q[i]

            self.assertEqual(len(q), q_len)

    def test_queue_threaded_stress(self) -> None:
        q: Queue = Queue()
        sub_num = 10
        views = [q.make_view() for _ in range(sub_num)]

        random.seed(83285020238474093)

        test_num = 10
        act_num = 100
        test_time = 0.5
        delay = 2 * test_time / test_num / act_num

        publisher_success = False
        subscriber_success = [False] * sub_num

        for _ in range(test_num):
            r_seq = [random.randint(0, 1_000_000_000) for _ in range(act_num)]

            publisher_success = False
            for i in range(sub_num):
                subscriber_success[i] = False

            def publisher(r_seq: list) -> None:
                nonlocal publisher_success
                for r in r_seq:
                    q.push(self.T(r))
                    sleep(random.uniform(0, delay))
                publisher_success = True

            def subscriber(r_seq: list, sub_ind: int, qw: QueueView) -> None:
                nonlocal subscriber_success
                for r in r_seq:
                    if len(qw) == 0:
                        while not qw.ready:
                            # if we use the following then the queue view is always of length 1
                            # qw.wait(random.uniform(0, delay))
                            # for stress testing we will let it accumulate a bit
                            sleep(random.uniform(0, delay))
                        qw.flush()
                    self.assertEqual(qw.top(), self.T(r))
                    with self.assertRaises(IndexError):
                        qw[qw.begin - 1]
                    with self.assertRaises(IndexError):
                        qw[qw.end]
                    self.assertLess(qw.begin, qw.end)
                    index = random.randint(qw.begin, qw.end - 1)
                    self.assertEqual(qw[index], self.T(r_seq[index]))
                    for el, ex in zip(qw, (r_seq[i] for i in range(qw.begin, qw.end)), strict=True):
                        self.assertEqual(el, self.T(ex))
                    self.assertEqual(qw.pop(), self.T(r))
                subscriber_success[sub_ind] = True

            threads = [Thread(target=publisher, args=(r_seq,))]
            for sub_ind in range(sub_num):
                threads.append(Thread(target=subscriber, args=(r_seq, sub_ind, views[sub_ind])))

            q.clear()

            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
            self.assertTrue(publisher_success and all(subscriber_success))


if __name__ == "__main__":
    unittest.main()
