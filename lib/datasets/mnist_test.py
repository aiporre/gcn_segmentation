from unittest import TestCase

import numpy as np

from .mnist import MNIST

data = MNIST('data/mnist', val_size=10000)


class MNISTTest(TestCase):
    def test_init(self):
        self.assertEqual(data.train.num_examples, 50000)
        self.assertEqual(data.val.num_examples, 10000)
        self.assertEqual(data.test.num_examples, 10000)

    def test_shapes(self):
        images, labels = data.train.next_batch(32, shuffle=False)
        self.assertEqual((32, 1, 28, 28), images.shape)
        self.assertEqual(labels.shape, (32, 28, 28))
        data.train.next_batch(data.train.num_examples - 32, shuffle=False)

        images, labels = data.val.next_batch(32, shuffle=False)
        self.assertEqual((32, 1, 28, 28), images.shape)
        self.assertEqual(labels.shape, (32, 28, 28))
        data.val.next_batch(data.val.num_examples - 32, shuffle=False)

        images, labels = data.test.next_batch(32, shuffle=False)
        self.assertEqual((32, 1, 28, 28), images.shape)
        self.assertEqual(labels.shape, (32, 28, 28))
        data.test.next_batch(data.test.num_examples - 32, shuffle=False)

    def test_images(self):
        images, _ = data.train.next_batch(
            data.train.num_examples, shuffle=False)

        self.assertEqual(images.dtype, np.float32)
        self.assertLessEqual(images.max(), 1)
        self.assertGreaterEqual(images.min(), 0)

        images, _ = data.val.next_batch(data.val.num_examples, shuffle=False)

        self.assertEqual(images.dtype, np.float32)
        self.assertLessEqual(images.max(), 1)
        self.assertGreaterEqual(images.min(), 0)

        images, _ = data.test.next_batch(data.test.num_examples, shuffle=False)

        self.assertEqual(images.dtype, np.float32)
        self.assertLessEqual(images.max(), 1)
        self.assertGreaterEqual(images.min(), 0)

    def test_labels(self):
        _, labels = data.train.next_batch(
            data.train.num_examples, shuffle=False)

        self.assertEqual(np.float, labels.dtype)

        _, labels = data.val.next_batch(
            data.val.num_examples, shuffle=False)

        self.assertEqual(np.float, labels.dtype)

        _, labels = data.test.next_batch(
            data.test.num_examples, shuffle=False)

        self.assertEqual(np.float, labels.dtype)

    def test_class_functions(self):
        self.assertEqual(['foreground', 'background'], data.classes)
        self.assertEqual(2, data.num_classes)

        _, labels = data.test.next_batch(5, shuffle=False)

        data.test.next_batch(data.test.num_examples - 5, shuffle=False)
