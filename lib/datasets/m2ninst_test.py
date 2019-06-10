from unittest import TestCase

import numpy as np
2
from .m2nist import M2NIST
data = M2NIST('data/M2NIST')


class M2NISTTest(TestCase):
    def test_init(self):
        self.assertEqual(data.train.num_examples, 4000)
        self.assertEqual(data.val.num_examples, 1000)
        self.assertEqual(data.test.num_examples, 1000)

    def test_shapes(self):
        images, labels = data.train.next_batch(32, shuffle=False)
        self.assertEqual((32, 1, 64, 84), images.shape)
        self.assertEqual((32, 64, 84), labels.shape)
        data.train.next_batch(data.train.num_examples - 32, shuffle=False)

        images, labels = data.val.next_batch(32, shuffle=False)
        self.assertEqual((32, 1, 64, 84), images.shape)
        self.assertEqual((32, 64, 84), labels.shape)
        data.val.next_batch(data.val.num_examples - 32, shuffle=False)
        
        images, labels = data.test.next_batch(32, shuffle=False)
        self.assertEqual((32, 1, 64, 84), images.shape)
        self.assertEqual((32, 64, 84), labels.shape)
        data.test.next_batch(data.test.num_examples - 32, shuffle=False)
    #
    # def test_images(self):
    #     images, _ = data.train.next_batch(
    #         data.train.num_examples, shuffle=False)
    #
    #     self.assertEqual(images.dtype, np.float32)
    #     self.assertLessEqual(images.max(), 1)
    #     self.assertGreaterEqual(images.min(), 0)
    #
    #     images, _ = data.val.next_batch(data.val.num_examples, shuffle=False)
    #
    #     self.assertEqual(images.dtype, np.float32)
    #     self.assertLessEqual(images.max(), 1)
    #     self.assertGreaterEqual(images.min(), 0)
    #
    #     images, _ = data.test.next_batch(data.test.num_examples, shuffle=False)
    #
    #     self.assertEqual(images.dtype, np.float32)
    #     self.assertLessEqual(images.max(), 1)
    #     self.assertGreaterEqual(images.min(), 0)
    #
    # def test_labels(self):
    #     _, labels = data.train.next_batch(
    #         data.train.num_examples, shuffle=False)
    #
    #     self.assertEqual(labels.dtype, np.uint8)
    #
    #     _, labels = data.val.next_batch(
    #         data.val.num_examples, shuffle=False)
    #
    #     self.assertEqual(labels.dtype, np.uint8)
    #
    #     _, labels = data.test.next_batch(
    #         data.test.num_examples, shuffle=False)
    #
    #     self.assertEqual(labels.dtype, np.uint8)
    #
    # def test_class_functions(self):
    #     self.assertEqual(data.classes,
    #                      ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    #     self.assertEqual(data.num_classes, 10)
    #
    #     _, labels = data.test.next_batch(5, shuffle=False)
    #
    #     self.assertEqual(data.classnames(labels[0]), ['7'])
    #     self.assertEqual(data.classnames(labels[1]), ['2'])
    #     self.assertEqual(data.classnames(labels[2]), ['1'])
    #     self.assertEqual(data.classnames(labels[3]), ['0'])
    #     self.assertEqual(data.classnames(labels[4]), ['4'])
    #
    #     data.test.next_batch(data.test.num_examples - 5, shuffle=False)
