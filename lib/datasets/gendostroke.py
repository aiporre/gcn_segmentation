import numpy as np
from scipy import ndimage

from .dataset import Datasets, Dataset
# CONSTANT WHERE TO FIND THE DATA
from config import ENDOSTROKE_DIR
import SimpleITK as sitk
import os
import pandas as pd
import os
from .dataset import Datasets, Dataset, GraphDataset
import torch
from torch_geometric.data import (Dataset, Data)
import torch_geometric.transforms as T
import numpy as np
from lib.graph import grid_tensor
from .download import maybe_download_and_extract
from imageio import imread
import nibabel as nib
TOTAL_SLICES = 1030
def load_nifti(filename):
    '''
    Reads nifti file
    '''
    img = nib.load(filename)
    data = img.get_fdata()
    return data



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
    return vessel_mask

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

def erode_mask(mask):
    return ndimage.binary_erosion(mask, structure=np.ones((1, 7, 7)))





class GENDOSTROKE(Datasets):
    def __init__(self, data_dir=ENDOSTROKE_DIR, batch_size=32, test_rate=0.2, annotated_slices=False, pre_transform=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_rate = test_rate
        if annotated_slices:
            train_dataset = _GENDOSTROKEA(self.data_dir, train=True, transform=T.Cartesian(), test_rate=test_rate,
                                       pre_transform=pre_transform)
            test_dataset = _GENDOSTROKEA(self.data_dir, train=False, transform=T.Cartesian(), test_rate=test_rate,
                                       pre_transform=pre_transform)
        else:
            train_dataset = _GENDOSTROKE(self.data_dir, train=True, transform=T.Cartesian(), test_rate=test_rate,
                                       pre_transform=pre_transform)
            test_dataset = _GENDOSTROKE(self.data_dir, train=False, transform=T.Cartesian(), test_rate=test_rate,
                                      pre_transform=pre_transform)

        train = GraphDataset(train_dataset, batch_size=self.batch_size, shuffle=True)
        test = GraphDataset(test_dataset, batch_size=self.batch_size, shuffle=False)

        super(GENDOSTROKE, self).__init__(train=train, test=test, val=test)

    @property
    def classes(self):
        return ['foreground', 'background']


class _GENDOSTROKE(Dataset):

    def __init__(self,
                 root,
                 train=True,
                 test_rate = 0.2,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.test_rate = test_rate
        self.train = train
        super(_GENDOSTROKE, self).__init__(root, transform, pre_transform,
                                         pre_filter)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        split = self.test_rate
        L = int(split*TOTAL_SLICES)
        if self.train:
            return ['gvessel_{:04d}.pt'.format(i) for i in range(TOTAL_SLICES-L)]
        else:
            return ['gvessel_{:04d}.pt'.format(i) for i in range(TOTAL_SLICES-L,TOTAL_SLICES)]

    def download(self):
        pass

    def __len__(self):
        return len(self.processed_file_names)

    def process(self):
        split = self.test_rate
        L = int(split*TOTAL_SLICES)
        max_slices = TOTAL_SLICES-L if self.train else L
        offset = 0 if self.train else TOTAL_SLICES-L
        cnt_slices = 0
        scan_i=20
        while cnt_slices<max_slices:
            scan_i+=1
            print('processed ', cnt_slices, ' out of ', max_slices)
            brain_mask = load_nifti(os.path.join(self.raw_dir, 'train', 'brain_masks', 'BM_{:02d}.nii'.format(scan_i)))
            ct_scan = np.load(os.path.join(self.raw_dir, 'train', 'scans', 'CT_A_{:02d}.nii'.format(scan_i)))

            brain_mask = brain_mask.astype(np.float)
            ct_scan = ct_scan.astype(np.float)

            ct_scan = (ct_scan-ct_scan.min())/(ct_scan.max()-ct_scan.min())
            ct_scan_masked = brain_mask*ct_scan
            # nz_slides = (ct_scan_masked.max(axis=(1,2))-ct_scan_masked.min(axis=(1,2))) != 0

            # ct_scan_masked = ct_scan_masked[nz_slides]
            stroke_mask = np.load(os.path.join(self.raw_dir, 'train', 'mask', 'SM_{:02d}.npy'.format(scan_i)))
            stroke_mask = stroke_mask*brain_mask

            usesful_scans = brain_mask.sum(axis=(1,2))>1000

            ct_scan_masked = ct_scan_masked[usesful_scans]
            stroke_mask = stroke_mask[usesful_scans]
            # process images and store them
            processed_num = len(stroke_mask) if cnt_slices+len(stroke_mask)<max_slices else max_slices-cnt_slices
            print('processing...: ' , processed_num)
            for i in range(processed_num):
                # print('---> file:', i+offset+cnt_slices)
                # Read data from `raw_path`.
                image = ct_scan_masked[i, :, :]
                mask = stroke_mask[i, :, :]
                if self.pre_transform is not None:
                    data = (image, mask)
                    data = self.pre_transform(data)
                else:
                    grid = grid_tensor((512, 512), connectivity=4)
                    grid.x = torch.tensor(image.reshape(512 * 512)).float()
                    grid.y = torch.tensor([mask.reshape(512 * 512)]).float()
                    data = grid

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue



                torch.save(data, os.path.join(self.processed_dir, 'gvessel_{:04d}.pt'.format(i+offset+cnt_slices)))

            # update counter
            cnt_slices+=processed_num



    def get(self, idx):
        # compute offset
        split = self.test_rate
        L = int(split*TOTAL_SLICES)
        offset = 0 if self.train else TOTAL_SLICES-L
        # get the file
        idx += offset
        data = torch.load(os.path.join(self.processed_dir, 'gvessel_{:04d}.pt'.format(idx)))
        return data




class _GENDOSTROKEA(Dataset):
    ''' Dataset GENDOSTROKEA
        Vessel12 selected and cleaned annotated slices
    '''

    def __init__(self,
                 root,
                 train=True,
                 test_rate = 0.2,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.test_rate = test_rate
        self.train = train
        super(_GENDOSTROKEA, self).__init__(root, transform, pre_transform,
                                         pre_filter)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        split = self.test_rate
        L = int(split*9)
        if self.train:
            return ['vessel_a_{}.pt'.format(i) for i in range(9-L)]
        else:
            return ['vessel_a_{}.pt'.format(i) for i in range(L,9)]

    def download(self):
        maybe_download_and_extract('https://transfer.sh/3xwmz/cured_vessel12.zip', self.raw_dir)

    def __len__(self):
        return len(self.processed_file_names)

    def process(self):
        self.download()
        # compute split
        split = self.test_rate
        L = int(split*9)
        # define range of scans to process
        scans_range = range(9-L) if self.train else range(L,9)
        # process scans
        for scan_i in scans_range:
            print('processed ', scan_i, ' out of ', len(scans_range))
            im = np.squeeze(imread(os.path.join(self.raw_dir, "lung{}.png".format(scan_i)))).astype(np.float)
            lb = np.squeeze(imread(os.path.join(self.raw_dir, "mask{}.png".format(scan_i)))).astype(np.float)
            mean = im.mean()
            std = im.std()
            im = (im-mean)/std
            lb = lb/lb.max()

            # Read data from `raw_path`.
            if self.pre_transform is not None:
                data = (im, lb)
                data = self.pre_transform(data)
            else:
                grid = grid_tensor((512, 512), connectivity=4)
                grid.x = torch.tensor(im.reshape(512*512)).float()
                grid.y = torch.tensor([lb.reshape(512*512)]).float()
                data = grid

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            torch.save(data, os.path.join(self.processed_dir, 'vessel_a_{}.pt'.format(scan_i)))



    def get(self, idx):
        # compute offset
        split = self.test_rate
        L = int(split*9)
        offset = 0 if self.train else L
        # get the file
        idx += offset
        data = torch.load(os.path.join(self.processed_dir, 'vessel_a_{}.pt'.format(idx)))
        return data



