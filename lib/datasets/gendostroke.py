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
from lib.process.progress_bar import printProgressBar
from .download import maybe_download_and_extract
from lib.utils.csv import csv_to_dict
from imageio import imread
import nibabel as nib
from pathlib import Path
TOTAL_SLICES = 4333
TOTAL_TRAIN_SLICES = 3432
TOTAL_TEST_SLICES = 901
NORMALIZED_SHAPE = {'Z': 158, 'Y': 189, 'X': 157}

def calculate_total():
    '''
    Runs one time to calculate total slice, then it is harcoded in the global variable
    '''
    total_slices = 0
    total_train = 0
    raw_dir = os.path.join(ENDOSTROKE_DIR,'raw')
    file_classes = csv_to_dict(os.path.join(raw_dir, 'splits.txt'), ',')

    for p in os.listdir(raw_dir):
        patient_path = os.path.join(raw_dir, p)
        patient_files = get_files_patient_path(patient_path)
        if not os.path.isdir(patient_path):
            continue
        if not patient_files["BRAIN"]:
            print(patient_path, ' has no files.')
        brain_mask = load_nifti(patient_files["BRAIN"][0], neurological_convension=True)
        brain_mask = brain_mask.astype(np.float)
        print('brain mask shape: ', brain_mask.shape)
        usesful_scans = brain_mask.sum(axis=(1, 2)) > 1000
        total_slices += np.sum(usesful_scans)
        if 'train' == file_classes[p]:
            total_train += np.sum(usesful_scans)
    return total_slices, total_train, total_slices - total_train

def load_nifti(filename, show_description=False, neurological_convension=False):
    '''
    Reads nifti file
    '''
    img = nib.load(filename)
    if show_description:
        print(' nifti file header: \n' + str(img.header))
    data = img.get_fdata()
    # if neurological convension then we transformt the data into Z,Y,X otherwise the X,Y, Z is keept
    if neurological_convension:
        data = np.transpose(data, (2,1,0))
    return data


def get_files_patient_path(patient_path):
    files = [f for f in Path(patient_path).rglob("*.nii") if f.name.startswith("Normalized")]
    patient_files = {}
    patient_files["CTN"] = list(filter(lambda f: "CT-N" in os.path.basename(f), files))
    patient_files["CTA"] = list(filter(lambda f: "CT-A" in os.path.basename(f), files))
    patient_files["CTP-MASK"] = list(filter(lambda f: "CT-P_mask" in os.path.basename(f), files))
    patient_files["CTP-TMAX"] = list(filter(lambda f: "CT-P_Tmax" in os.path.basename(f), files))
    patient_files["CTP-CBF"] = list(filter(lambda f: "CT-P_CBF" in os.path.basename(f), files))
    patient_files["CTP-CBV"] = list(filter(lambda f: "CT-P_CBV" in os.path.basename(f), files))
    patient_files["CTP-RAW"] = list(filter(lambda f: "CT-P_raw" in os.path.basename(f), files))
    patient_files["LESION"] = list(filter(lambda f: "Lesion_" in os.path.basename(f), files))
    # now look for the generated masks

    files = [f for f in Path(patient_path).rglob("*.nii.gz") if f.name.startswith("Normalized")]
    patient_files["BRAIN"] = list(
        filter(lambda f: "CT-N" in os.path.basename(f) and "nii_mask" in os.path.basename(f), files))
    # whe mode that one lession select the last one:
    # if len(patient_files["LESION"]) >1:
    #       dates = map(patient_files["LESION"])
    return patient_files


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
        print('Warning: In the ENDOSTROKE dataset, the test/train rate is predefined by the split file.')
        self.test_rate = 1
        self.L = TOTAL_TRAIN_SLICES
        self.train = train
        super(_GENDOSTROKE, self).__init__(root, transform, pre_transform,
                                         pre_filter)

    @property
    def raw_file_names(self):
        files = []
        file_classes = csv_to_dict(os.path.join(self.raw_dir, 'splits.txt'),',')
        if self.train:
            for f, c in file_classes.items():
                if c == 'train':
                    files.append(f)
        else:
            for f, c in file_classes.items():
                if c == 'val':
                    files.append(f)
        return files

    @property
    def processed_file_names(self):
        split = self.test_rate
        L = self.L #int(split*TOTAL_SLICES)
        if self.train:
            return ['gendo_{:04d}.pt'.format(i) for i in range(L)]
        else:
            return ['gendo_{:04d}.pt'.format(i) for i in range(L,TOTAL_SLICES)]

    def download(self):
        pass

    def __len__(self):
        return len(self.processed_file_names)

    def process(self):
        split = self.test_rate
        L = self.L # int(split*TOTAL_SLICES)
        max_slices = L if self.train else TOTAL_SLICES-L
        offset = 0 if self.train else L
        cnt_slices = 0
        scan_i=0
        progressBarPrefix = f'Generating samples for dataset {self.__class__} train={self.train}'
        printProgressBar(0, max_slices, prefix=progressBarPrefix)
        while cnt_slices<max_slices:
            print('processed ', cnt_slices, ' out of ', max_slices)
            patient_files = get_files_patient_path(self.raw_paths[scan_i])
            scan_i+=1
            brain_mask = load_nifti(patient_files["BRAIN"][0], neurological_convension=True)
            ct_scan = load_nifti(patient_files["CTA"][0], neurological_convension=True)

            brain_mask = brain_mask.astype(np.float)
            ct_scan = ct_scan.astype(np.float)

            ct_scan = (ct_scan-ct_scan.min())/(ct_scan.max()-ct_scan.min())
            ct_scan_masked = brain_mask*ct_scan
            # nz_slides = (ct_scan_masked.max(axis=(1,2))-ct_scan_masked.min(axis=(1,2))) != 0

            # ct_scan_masked = ct_scan_masked[nz_slides]
            lession_files = patient_files['LESION']
            stroke_masks = [load_nifti(ff, neurological_convension=True) for ff in lession_files]
            stroke_mask = np.ones_like(stroke_masks[0])
            for sm in stroke_masks:
               stroke_mask =  sm*stroke_mask

            stroke_mask = stroke_mask*brain_mask

            usesful_scans = brain_mask.sum(axis=(1,2))>1000

            ct_scan_masked = ct_scan_masked[usesful_scans]
            stroke_mask = stroke_mask[usesful_scans]
            # process images and store them
            processed_num = len(stroke_mask) if cnt_slices+len(stroke_mask)<max_slices else max_slices-cnt_slices
            print('processing...: ' , processed_num)
            for i in range(processed_num):
                #print('---> file:', i+offset+cnt_slices)
                printProgressBar(i+cnt_slices, max_slices, prefix=progressBarPrefix, suffix=f'sample={i+offset+cnt_slices}')
                # Read data from `raw_path`.
                image = ct_scan_masked[i, :, :]
                mask = stroke_mask[i, :, :]
                if self.pre_transform is not None:
                    data = (image, mask)
                    data = self.pre_transform(data)
                else:
                    grid = grid_tensor((NORMALIZED_SHAPE['Y'], NORMALIZED_SHAPE['X']), connectivity=4)
                    num_elements = NORMALIZED_SHAPE['Y'] * NORMALIZED_SHAPE['X']
                    grid.x = torch.tensor(image.reshape(num_elements)).float()
                    grid.y = torch.tensor([mask.reshape(num_elements)]).float()
                    data = grid

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                torch.save(data, os.path.join(self.processed_dir, 'gendo_{:04d}.pt'.format(i+offset+cnt_slices)))

            # update counter
            cnt_slices+=processed_num



    def get(self, idx):
        # compute offset
        L = self.L #int(split*TOTAL_SLICES)
        offset = 0 if self.train else L
        # get the file
        idx += offset
        data = torch.load(os.path.join(self.processed_dir, 'gendo_{:04d}.pt'.format(idx)))
        return data





