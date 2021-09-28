import os
from unittest import TestCase
from config import ENDOSTROKE_DIR
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .gendostroke import load_nifti, get_files_patient_path, calculate_total


class LoadingDataTest(TestCase):
    def test_load_nifti(self):
        files = [f for f in Path(ENDOSTROKE_DIR).rglob("*.nii") if f.name.startswith("Normalized")]
        print(files)
        filename = files[0]
        print("Reading file: ", filename)
        window_center = 30
        window_width = 60

        X = load_nifti(filename, show_description=True)
        # X = load_nifti(filename)
        X = X.astype(float)
        # X = X*(X>0)
        # X = (X-X.min())/(X.max()-X.min())

        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        # X = np.clip(X, img_min, img_max)
        X = X * (X > img_min) * (X < img_max)

        plt.figure()
        i = 0
        plt.imshow(X[:, :, i].squeeze())
        plt.title("slice 100 input " + str(i))
        for i in range(X.shape[-1]):
            plt.subplot(8, 2, i + 1)
            plt.imshow(X[:, :, i].squeeze())
            plt.title("slice 100 input " + str(i))
        plt.colorbar()
        plt.show()
        print(X.max())
        print(X.min())

    def test_get_patient_files(self):
        patient_dir = os.path.join(ENDOSTROKE_DIR, "1245")
        print('searching in : ', patient_dir)
        patients_files = get_files_patient_path(patient_dir)
        for k, v in patients_files.items():
            print('===> k = ', k)
            print('===> v = ', v)

    def test_calculate_total(self):
        max_slices, train_slices, val_slices = calculate_total()
        print('TOTAL NUMBER OF SLICES: ', max_slices)
        print('TRAIN SLICES: ', train_slices)
        print('VAL SLICES: ', val_slices)
