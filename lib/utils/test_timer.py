from unittest import TestCase
import os
from .timer import Timer

class TestTimer(TestCase):
    def test_init(self):
        t = Timer(100)
        self.assertEqual(t.time_threshold,100)
    def test_time(self):
        t = Timer(10)
        t1 = t.start
        if t.is_time():
            self.assertNotEqual(t.start, t1)


