import os
from unittest import TestCase
from .gisles2018 import get_files_patient_path, load_nifti, _ISLESFoldIndices
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


class Test_ISLESFoldIndices(TestCase):
    def setUp(self):
        split_text = ["CASES,fold_1,fold_2,fold_3,fold_4,fold_5",
                      "case_1,train,val,train,test,train",
                      "case_10,train,test,train,train,test",
                      "case_11,train,test,train,train,train",
                      "case_12,train,train,train,train,train",
                      "case_13,train,train,train,train,train",
                      "case_14,train,train,train,train,train",
                      "case_15,train,val,test,train,test",
                      "case_16,test,test,test,train,val",
                      "case_17,train,train,val,train,test",
                      "case_18,train,train,test,train,test"]
        self.split_file= "./split.txt"
        with open(self.split_file, "w") as f:
            f.write("\n".join(split_text)+ "\n")
        self.processed_mapping_file = "processed_mapping.txt"
        text_processed_files = ["case_1,0,True",
                                "case_1,1,True",
                                "case_1,2,True",
                                "case_1,3,True",
                                "case_1,4,True",
                                "case_1,5,True",
                                "case_1,6,True",
                                "case_1,7,True",
                                "case_10,8,True",
                                "case_10,9,True",
                                "case_10,10,True",
                                "case_10,11,True",
                                "case_11,12,True",
                                "case_11,13,True",
                                "case_11,14,True",
                                "case_11,15,True",
                                "case_12,16,True",
                                "case_12,17,True",
                                "case_12,18,True",
                                "case_12,19,True",
                                "case_13,20,True",
                                "case_13,21,True",
                                "case_13,22,True",
                                "case_13,23,True",
                                "case_14,24,True",
                                "case_14,25,True",
                                "case_14,26,True",
                                "case_14,27,True",
                                "case_15,28,True",
                                "case_15,29,True",
                                "case_15,30,True",
                                "case_15,31,True",
                                "case_17,32,True",
                                "case_17,33,True",
                                "case_17,34,True",
                                "case_17,35,True",
                                "case_18,36,True",
                                "case_18,37,False"]
        with open(self.processed_mapping_file, "w") as f:
            f.writelines("\n".join(text_processed_files)+"\n")

    def test_get_cases(self):
        cases = ["case_1", "case_10", "case_11", "case_12", "case_13",
                 "case_14", "case_15", "case_17", "case_18"]
        indices = _ISLESFoldIndices(self.processed_mapping_file, 1, "train")
        print(indices.get_cases())
        self.assertTrue(all([c in indices.get_cases() for c in cases]))
        self.assertTrue(all([c in cases for c in indices.get_cases()]))
        x = indices.get_by_case_id("case_18")
        self.assertEqual(x, [36, 37])
        x = indices.get_by_case_id("case_18", useful=True)
        self.assertEqual(x, [36])

    def tearDown(self) -> None:
        os.remove(self.processed_mapping_file)
        os.remove(self.split_file)
