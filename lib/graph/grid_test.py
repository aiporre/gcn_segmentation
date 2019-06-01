from unittest import TestCase

from numpy.testing import assert_equal

from .grid import grid_adj, grid_points, grid_mass, grid_tensor

from torch_geometric.transforms import Cartesian

class GridTest(TestCase):
    def test_grid_points(self):
        points = grid_points((3, 2))
        expected = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
        assert_equal(points, expected)

    def test_grid_mass(self):
        mass = grid_mass((3, 2))
        expected = [1, 1, 1, 1, 1, 1]
        assert_equal(mass, expected)

    def test_grid_adj_connectivity_4(self):
        adj = grid_adj((3, 2), connectivity=4)
        expected = [[0, 1, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 1, 0],
                    [0, 1, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 1, 0]]

        assert_equal(adj.toarray(), expected)

    def test_grid_adj_connectivity_8(self):
        adj = grid_adj((3, 2), connectivity=8)
        expected = [[0, 1, 1, 1, 0, 0], [1, 0, 1, 1, 0, 0], [1, 1, 0, 1, 1, 1],
                    [1, 1, 1, 0, 1, 1], [0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 1, 0]]

        assert_equal(adj.toarray(), expected)

    def test_grid_tensor(self):
        tensor = grid_tensor((189,157), connectivity=4)
        print('grid tensor: ', tensor)
        cartesian_transform  = Cartesian()
        cartesian_tensor = cartesian_transform(tensor)
        print('Cartesian tensor: ', cartesian_tensor)

        tensor = grid_tensor((189, 157), connectivity=8)
        print('grid tensor: ', tensor)
        cartesian_transform = Cartesian()
        cartesian_tensor = cartesian_transform(tensor)
        print('Cartesian tensor: ', cartesian_tensor)
