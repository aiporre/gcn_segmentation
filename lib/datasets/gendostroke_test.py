from unittest import TestCase
from config import ENDOSTROKE_DIR
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .gendostroke import load_nifti

class LoadingDataTest(TestCase):
    def test_load_nifti(self):
        filename = [f for f in Path(ENDOSTROKE_DIR).glob("*.nii")][1]
        window_center = 30
        window_width = 60

        X = load_nifti(filename)
        X = X.astype(float)
        # X = X*(X>0)
        # X = (X-X.min())/(X.max()-X.min())

        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        # X = np.clip(X, img_min, img_max)
        X = X *(X>img_min)*(X<img_max)


        plt.figure()
        i = 0
        plt.imshow(X[:,:,100,i].squeeze())
        plt.title("slice 100 input " + str(i))
        # for i in range(X.shape[-1]):
        #     plt.subplot(8,2,i+1)
        #     plt.imshow(X[:,:,100,i].squeeze())
        #     plt.title("slice 100 input " + str(i))
        plt.colorbar()
        plt.show()
        print(X.max())
        print(X.min())



