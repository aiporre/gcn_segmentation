import numpy as np
from .dataset import Datasets, Dataset
# CONSTANT WHERE TO FIND THE DATA
from config import VESSEL_DIR
import SimpleITK as sitk
import os
from .dataset import Datasets, Dataset, GraphDataset
import torch
from torch_geometric.data import (Dataset, Data)
import torch_geometric.transforms as T
import numpy as np
from lib.graph import grid_tensor

from .vessel_synth import read_dataset_mhd

TOTAL_SLICES = 5050



def load_itk(filename):
    ''' Reads scan with coordinates frame Z,Y,X with origin at '''
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing

def load_vessel_mask_pre(image, threshold=0):
    vessel_mask = image > threshold
    return vessel_mask.astype(np.float)

class GSVESSEL(Datasets):
    def __init__(self, data_dir=VESSEL_DIR, batch_size=32, test_rate=0.2, annotated_slices=False, pre_transform=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_rate = test_rate

        train_dataset = _GSVESSEL(self.data_dir, train=True, transform=T.Cartesian(), test_rate=test_rate,
                                   pre_transform=pre_transform)
        test_dataset = _GSVESSEL(self.data_dir, train=False, transform=T.Cartesian(), test_rate=test_rate,
                                  pre_transform=pre_transform)

        train = GraphDataset(train_dataset, batch_size=self.batch_size, shuffle=True)
        test = GraphDataset(test_dataset, batch_size=self.batch_size, shuffle=False)

        super(GSVESSEL, self).__init__(train=train, test=test, val=test)



class _GSVESSEL(Dataset):

    def __init__(self,
                 root,
                 train=True,
                 test_rate = 0.2,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.test_rate = test_rate
        self.train = train
        super(_GSVESSEL, self).__init__(root, transform, pre_transform,
                                         pre_filter)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        split = self.test_rate
        L = int(split*TOTAL_SLICES)
        if self.train:
            return ['data_{:04d}.pt'.format(i) for i in range(TOTAL_SLICES-L)]
        else:
            return ['data_{:04d}.pt'.format(i) for i in range(TOTAL_SLICES-L,TOTAL_SLICES)]

    def download(self):
        pass

    def __len__(self):
        return len(self.processed_file_names)

    def process(self):
        split = self.test_rate
        L = int(split*TOTAL_SLICES)
        max_slices = TOTAL_SLICES-L if self.train else L
        offset = 0 if self.train else TOTAL_SLICES-L
        vessel_data = read_dataset_mhd(self.raw_dir)
        vessel_data = vessel_data['train'] if self.train else vessel_data['test']

        for i in range(max_slices):
            print('processed ', i, ' out of ', max_slices)
            image = vessel_data['images'][i, 0, :, :]
            mask = vessel_data['labels'][i, :, :]
            if self.pre_transform is not None:
                data = (image, mask)
                data = self.pre_transform(data)
            else:
                grid = grid_tensor((101, 101), connectivity=4)
                grid.x = torch.tensor(image.reshape(101 * 101)).float()
                grid.y = torch.tensor([mask.reshape(101 * 101)]).float()
                data = grid

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            torch.save(data, os.path.join(self.processed_dir, 'data_{:04d}.pt'.format(i+offset)))




    def get(self, idx):
        # compute offset
        split = self.test_rate
        L = int(split*TOTAL_SLICES)
        offset = 0 if self.train else TOTAL_SLICES-L
        # get the file
        idx += offset
        data = torch.load(os.path.join(self.processed_dir, 'data_{:04d}.pt'.format(idx)))
        return data




