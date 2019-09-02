from unittest import TestCase
from .transforms import Crop
import numpy as np

class TestTransfroms(TestCase):
    def test_crop(self):
        xx = np.linspace(0, 1, 512)
        yy = np.linspace(0, 1, 512)
        X,Y = np.meshgrid(xx,yy)
        Z = X**2+Y**2
        crop = Crop(100,100,50,50,graph_mode=False)
        Z_transformed = crop((Z,Z))

