from unittest import TestCase
from .files import upload_training, get_npy_files

class LoggerTest(TestCase):
    def test_print_debug(self):
        upload_training('UNet',1,0.001,'MNIST')
