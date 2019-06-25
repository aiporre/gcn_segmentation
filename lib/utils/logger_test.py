from unittest import TestCase
from .logger import print_debug
import config as c
class LoggerTest(TestCase):
    def test_print_debug(self):
        c.DEBUG = False
        print_debug("this message should not appear")
        c.DEBUG = True
        print_debug("This a debug message")
        try:
            raise RuntimeError
        except RuntimeError as e:
            print_debug("this is runtime error/exception",exception=e)
