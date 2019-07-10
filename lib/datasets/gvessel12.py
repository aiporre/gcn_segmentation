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
        split = self.test_rate
        L = int(split*1325)
        if self.train:
            return ['data_{:04d}.pt'.format(i) for i in range(1325-L)]
        else:
            return ['data_{:04d}.pt'.format(i) for i in range(L,1325)]

    def download(self):
        pass

    def __len__(self):
        return len(self.processed_file_names)

    def process(self):
        split = self.test_rate
        L = int(split*1325)
        max_slices = 1325-L if self.train else L
        offset = 0 if self.train else L
        cnt_slices = 0
        scan_i=20
        while cnt_slices<max_slices:
            scan_i+=1
            print('processed ', cnt_slices, ' out of ', max_slices)
            lung_mask, _, _ = load_itk(os.path.join(self.raw_dir, 'train', 'Lungmasks', 'VESSEL12_{:02d}.mhd'.format(scan_i)))
            ct_scan, origin, spacing = load_itk(
                os.path.join(self.raw_dir, 'train', 'Scans', 'VESSEL12_{:02d}.mhd'.format(scan_i)))
            ct_scan_masked = lung_mask*ct_scan
            ct_scan_masked.astype(np.float, copy=False)
            ct_scan_masked = (ct_scan_masked-ct_scan_masked.min())/(ct_scan_masked.max()-ct_scan_masked.min())

            vessel_mask, _ = load_vessel_mask_pre(ct_scan.shape, os.path.join(self.raw_dir, 'train', 'Annotations',
                                                                              'VESSEL12_{:02d}_OutputVolume.npy'.format(
                                                                                  scan_i)))

            processed_num = len(ct_scan) if cnt_slices+len(ct_scan)<max_slices else max_slices-cnt_slices
            cnt_slices+=processed_num


            for i in range(processed_num):
                # Read data from `raw_path`.
                grid = grid_tensor((512, 512), connectivity=4)
                grid.x = torch.tensor(ct_scan_masked[i, :, :].reshape(512 * 512)).float()
                grid.y = torch.tensor([vessel_mask[i, :, :].reshape(512 * 512)]).float()
                data = grid

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, os.path.join(self.processed_dir, 'data_{:03d}.pt'.format(i+offset+cnt_slices)))


    def get(self, idx):
        # compute offset
        split = self.test_rate
        L = int(split*1325)
        offset = 0 if self.train else L
        # get the file
        idx += offset
        data = torch.load(os.path.join(self.processed_dir, 'data_{:04d}.pt'.format(idx)))
        return data

