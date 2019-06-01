from unittest import TestCase
from .savefigs import savefigs
import numpy as np
import matplotlib.pyplot as plt



class TestSaveFigs(TestCase):
    def test_savefigs(self):
        delta = 0.025
        x = y = np.arange(-3.0, 3.0, delta)
        X, Y = np.meshgrid(x, y)
        Z1 = np.exp(-X ** 2-Y ** 2)
        Z2 = np.exp(-(X-1) ** 2-(Y-1) ** 2)
        Z = (Z1-Z2)*2
        fig = plt.figure()
        ax = plt.subplot(1,1,1)
        ax.imshow(Z)
        savefigs('test_savefigs.png','./figs/test_savefigs/',fig)


