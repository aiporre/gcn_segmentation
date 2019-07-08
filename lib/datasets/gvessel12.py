import numpy as np
from .dataset import Datasets, Dataset
# CONSTANT WHERE TO FIND THE DATA
from config import VESSEL_DIR
import SimpleITK as sitk
import os
import pandas as pd
import os
from .dataset import Datasets, Dataset, GraphDataset
import torch
from torch_geometric.data import (Dataset, Data)
import torch_geometric.transforms as T
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from lib.graph import grid_tensor



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


def load_vessel_mask_pre(shape, path):
    vessel_mask = np.load(os.path.join(path))
    return vessel_mask, range(200,205)

def load_vessel_mask_csv(shape, path):
    ''' Reads annotation csv and produces the vessel mask with coordinates Z,Y,X'''
    df = pd.read_csv(path, sep=',', header=None, names=['x','y','z','annotation'])
    x, y, z, annotations = df.x.values , df.y.values, df.z.values, df.annotation
    vessel_mask = np.zeros(shape,dtype=np.float)
    vessel_mask[z,y,x] = 2*annotations-1
    # print('z',z,'\ny',y,'\nx',x)
    # unique list of z annotated slices
    z_slices = np.unique(z)
    return vessel_mask, z_slices


def read_dataset(data_dir, test_rate):
    '''
        Reads the directory and conforms the structure of generic datasets:
        {'train': {'images': list of images, 'labels': list of labels}
         'test': {'images': list of images, 'labels': list of labels}}
    '''
    output = {'train': {'images':[], 'labels':[]} , 'test': {'images':[], 'labels':[]}}
    images = []
    labels = []
    for i in [21,22,23]:
        # reading the ct-scan masked with the lungs
        lung_mask, _, _ = load_itk(os.path.join(data_dir, 'train', 'Lungmasks', 'VESSEL12_{:02d}.mhd'.format(i)))
        ct_scan, origin, spacing = load_itk(os.path.join(data_dir, 'train', 'Scans', 'VESSEL12_{:02d}.mhd'.format(i)))
        ct_scan_masked = lung_mask*ct_scan
        ct_scan_masked.astype(np.float,copy=False)
        ct_scan_masked = (ct_scan_masked-ct_scan_masked.min())/(ct_scan_masked.max()-ct_scan_masked.min())

        vessel_mask, _ = load_vessel_mask_pre(ct_scan.shape, os.path.join(data_dir, 'train', 'Annotations', 'VESSEL12_{:02d}_OutputVolume.npy'.format(i)))
        # alternatively, we may curate the 9 slices there fore we need to know which slices were annotated
        # vessel_mask_annotations, z_slices = load_vessel_mask_csv(ct_scan.shape, os.path.join(data_dir, 'train', 'Annotations', 'VESSEL12_{:02d}_Annotations.csv'.format(i)))
        images += [ct_scan_masked[i,:,:] for i in range(len(ct_scan_masked))]
        labels += [vessel_mask[i, :, :] for i in range(len(vessel_mask))]
    # TODO: split is hardcoded
    split = test_rate
    L = int(split*len(images))
    output['train']['images'], output['test']['images'] = np.stack(images[:-L], axis=0), np.stack(images[-L:], axis=0)
    output['train']['labels'], output['test']['labels'] = np.stack(labels[:-L], axis=0), np.stack(labels[-L:], axis=0)
    return output



class GVESSEL12(Datasets):
    def __init__(self, data_dir=VESSEL_DIR, batch_size=32, test_rate=0.2):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_rate = test_rate

        train_dataset = _GVESSEL12(self.data_dir, train=True, transform=T.Cartesian())
        test_dataset = _GVESSEL12(self.data_dir, False, transform=T.Cartesian())

        train = GraphDataset(train_dataset, batch_size=self.batch_size, shuffle=True)
        test = GraphDataset(test_dataset, batch_size=self.batch_size, shuffle=False)

        super(GVESSEL12, self).__init__(train=train, test=test, val=test)



class _GVESSEL12(Dataset):

    def __init__(self,
                 root,
                 train=True,
                 test_rate = 0.2,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.test_rate = test_rate
        self.train = train
        super(_GVESSEL12, self).__init__(root, transform, pre_transform,
                                         pre_filter)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        if self.train:
            return ['data_{}.pt'.format(i) for i in range(8000)]
        else:
            return ['data_{}.pt'.format(i) for i in range(8000,10000)]

    def download(self):
        pass

    def __len__(self):
        return len(self.processed_file_names)

    def process(self):
        i = self.offset
        vessel12 = read_dataset(self.raw_dir)
        images = vessel12['train']['images'] if self.train else vessel12['test']['images']
        masks = vessel12['train']['labels'] if self.train else vessel12['test']['images']

        for image, mask in zip(images, masks):
            # Read data from `raw_path`.
            grid = grid_tensor((28, 28), connectivity=4)
            grid.x = torch.tensor(image.reshape(28 * 28)).float()
            grid.y = torch.tensor([mask.reshape(28 * 28)]).float()
            data = grid

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))
            i += 1

    def get(self, idx):
        idx += self.offset
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data

