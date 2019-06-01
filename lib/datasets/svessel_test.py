from unittest import TestCase

import numpy as np

from .svessel import SVESSEL


data = SVESSEL()


class SVESSELTest(TestCase):
    def test_init(self):
        self.assertEqual(4040, data.train.num_examples)
        self.assertEqual(1010, data.val.num_examples)
        self.assertEqual(1010, data.test.num_examples)

    def test_shapes(self):
        images, labels = data.train.next_batch(32, shuffle=False)
        self.assertEqual((32, 1, 101, 101), images.shape)
        self.assertEqual(labels.shape, (32, 101, 101))
        data.train.next_batch(data.train.num_examples - 32, shuffle=False)

        images, labels = data.val.next_batch(32, shuffle=False)
        self.assertEqual((32, 1, 101, 101), images.shape)
        self.assertEqual(labels.shape, (32, 101, 101))
        data.val.next_batch(data.val.num_examples - 32, shuffle=False)

        images, labels = data.test.next_batch(32, shuffle=False)
        self.assertEqual((32, 1, 101, 101), images.shape)
        self.assertEqual(labels.shape, (32, 101, 101))
        data.test.next_batch(data.test.num_examples - 32, shuffle=False)

    def test_images(self):
        images, _ = data.train.next_batch(
            data.train.num_examples, shuffle=False)

        self.assertEqual(np.float, images.dtype)
        self.assertLessEqual(images.max(), 1)
        self.assertGreaterEqual(images.min(), 0)

        images, _ = data.val.next_batch(data.val.num_examples, shuffle=False)

        self.assertEqual(np.float, images.dtype)
        self.assertLessEqual(images.max(), 1)
        self.assertGreaterEqual(images.min(), 0)

        images, _ = data.test.next_batch(data.test.num_examples, shuffle=False)

        self.assertEqual(np.float, images.dtype)
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


