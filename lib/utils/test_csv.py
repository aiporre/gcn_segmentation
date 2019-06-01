from unittest import TestCase
from .csv import csv_to_dict
import io


def create_split_file():
    # TODO: create a mock of a split.txt file. Check here:
    data = {'1000':'train', "1021":"val"}
    # this should go into a mock file that is used in the utils.csv_to_dict
    pass

class Test(TestCase):
    def test_csv_to_dict(self):
        self.fail()
