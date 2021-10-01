from unittest import TestCase
from .gisles2018 import get_files_patient_path
class DatasetGILSES2018Test(TestCase):
    def test_get_files_patient_path(self):
        patient_files = get_files_patient_path("case_1")
        for i, k in patient_files.items():
            print('CASE: ', i)
            print('iterms: ', k)
