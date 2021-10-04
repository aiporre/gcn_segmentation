import os
from unittest import TestCase
from .gisles2018 import get_files_patient_path, load_nifti
from config import ISLES2018_DIR


class DatasetGILSES2018Test(TestCase):
    def test_get_files_patient_path(self):
        patient_path = os.path.join(ISLES2018_DIR, "raw", "TRAINING", "case_1")
        print("LOOKING IN PATH: ", patient_path)
        patient_files = get_files_patient_path(patient_path)
        for i, k in patient_files.items():
            print('CASE: ', i)
            print('iterms: ', k)

    def test_load_nifti(self):
        patient_path = os.path.join(ISLES2018_DIR, "raw", "TRAINING", "case_1")
        patient_files = get_files_patient_path(patient_path)
        print(patient_files)
        ct_scan_path = patient_files['CTN'][0]
        data = load_nifti(ct_scan_path, show_description=True, neurological_convension=True)
