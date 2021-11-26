import csv

import numpy as np
from scipy import ndimage

from .dataset import Datasets, Dataset
# CONSTANT WHERE TO FIND THE DATA
from config import ISLES2018_DIR
import SimpleITK as sitk
import os
import pandas as pd
import os
from .dataset import Datasets, Dataset, GraphDataset
import torch
from torch_geometric.data import (Dataset, Data )
from torch_geometric.data.makedirs import makedirs
import torch_geometric.transforms as T
import numpy as np
from lib.graph import grid_tensor
from lib.process.progress_bar import printProgressBar
from .download import maybe_download_and_extract
from lib.utils.csv import csv_to_dict
from imageio import imread
import nibabel as nib
from pathlib import Path

NORMALIZED_SHAPE = {'Z': None, 'Y': 256, 'X': 256}

def get_modalities(arg_mod):
    modalities = {"CTN": "CTN", "TMAX": "CTP-TMAX", "CBF": "CTP-CBF", "CBV": "CTP-CBV", "MTT": "CTP-MTT"}
    output_modalities = [modalities[mod] for mod in arg_mod]
    return output_modalities


def isles2018_reshape(x):
    '''
    Transform used by plotting functions
    '''
    N = x.numel() if isinstance(x, torch.Tensor) else x.size
    nn = NORMALIZED_SHAPE['Y'] * NORMALIZED_SHAPE['X']
    if N == nn:
        isles_shape = (NORMALIZED_SHAPE['Y'], NORMALIZED_SHAPE['X'])
    else:
        N_channels = N // nn
        isles_shape = (NORMALIZED_SHAPE['Y'], NORMALIZED_SHAPE['X'], N_channels)
    if isinstance(x, torch.Tensor):
        x = torch.reshape(x, isles_shape)
    else:
        x = np.reshape(x, isles_shape)
    if N != nn:
        x = x[...,0]
    return x

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


def get_files_patient_path(patient_path, target='training'):
    files = list(Path(patient_path).rglob("*.nii"))
    patient_files = {}
    patient_files["CTN"] = list(filter(lambda f: "CT." in os.path.basename(f), files))
    patient_files["CTP-TMAX"] = list(filter(lambda f: "CT_Tmax." in os.path.basename(f), files))
    patient_files["CTP-CBF"] = list(filter(lambda f: "CT_CBF." in os.path.basename(f), files))
    patient_files["CTP-CBV"] = list(filter(lambda f: "CT_CBV." in os.path.basename(f), files))
    patient_files["CTP-MTT"] = list(filter(lambda f: "CT_MTT." in os.path.basename(f), files))
    patient_files["LESION"] = list(filter(lambda f: "OT." in os.path.basename(f), files))
    return patient_files


def erode_mask(mask):
    return ndimage.binary_erosion(mask, structure=np.ones((1, 7, 7)))


class GISLES2018(Datasets):
    def __init__(self, data_dir=ISLES2018_DIR,
                 batch_size=32,
                 test_rate=0.2,
                 annotated_slices=False,
                 pre_transform=None,
                 fold=1,
                 modalities=("CTN", "CTP-TMAX", "CTP-CBF", "CTP-CBV", "CTP-MTT")):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_rate = test_rate
        train_dataset = _GISLES2018(self.data_dir, dataset_type='train', transform=T.Cartesian(),
                                   pre_transform=pre_transform, fold=fold, modalities=modalities)
        val_dataset = _GISLES2018(self.data_dir, dataset_type='val', transform=T.Cartesian(),
                                   pre_transform=pre_transform, fold=fold, modalities=modalities)
        test_dataset = _GISLES2018(self.data_dir, dataset_type='test', transform=T.Cartesian(),
                                  pre_transform=pre_transform, fold=fold, modalities=modalities)

        train = GraphDataset(train_dataset, batch_size=self.batch_size, shuffle=True)
        val = GraphDataset(val_dataset, batch_size=self.batch_size, shuffle=False)
        test = GraphDataset(test_dataset, batch_size=self.batch_size, shuffle=False)
        super(GISLES2018, self).__init__(train=train, test=test, val=val)

    @property
    def classes(self):
        return ['foreground', 'background']


class _GISLES2018(Dataset):

    def __init__(self,
                 root,
                 transform=None,
                 dataset_type='train',
                 pre_transform=None,
                 pre_filter=None,
                 fold=1,
                 split_dir="TRAINING", 
                 modalities=("CTN", "CTP-TMAX", "CTP-CBF", "CTP-CBV", "CTP-MTT")):
        print('Warning: In the ISLES2018 dataset, the test/train rate is predefined by file distribution.')
        self.root = root
        self.test_rate = 1
        self.fold = fold
        self.dataset_type = dataset_type
        self.split_dir = split_dir
        self.modalities = modalities
        # creates the processed dir

        raw_dir = os.path.join(self.root, 'raw', self.split_dir)
        processed_dir = os.path.join(self.root, 'processed', self.split_dir)
        makedirs(processed_dir)
        # procesed mapping is csv file that tell which proccesd files match with which case_XY e.g. gilses_1000 --> case_10
        self.indices = _ISLESFoldIndices(cache_file=os.path.join(root, 'raw', self.split_dir, 'processed_mapping.txt'),
                                         fold=fold, dataset_type=dataset_type)
        super(_GISLES2018, self).__init__(root, transform, pre_transform, pre_filter)
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir

    # @property
    # def raw_dir(self):
    #     print(' reassign raw path')
    #     raw_path = os.path.join(self.root, 'raw', self.split_dir)
    #     return raw_path
    #
    # @property
    # def processed_dir(self):
    #     print(' reassign processed path')
    #     processed_path = os.path.join(self.root, 'processed', self.split_dir)
    #     return processed_path

    @property
    def raw_file_names(self):
        files = self.indices.get_cases()
        return files

    @property
    def processed_file_names(self):
        processed_files = []
        for case_id in self.raw_file_names:
            processed_indices = self.indices.get_by_case_id(case_id)
            _processed_files = ['gendo_{:04d}.pt'.format(i) for i in processed_indices]
            processed_files.extend(_processed_files)
        return processed_files

    @property
    def raw_paths(self):
        r"""The filepaths to find in order to skip the download."""
        files = self.raw_file_names
        return [os.path.join(self.root, 'raw', self.split_dir, f) for f in files]

    @property
    def processed_paths(self):
        r"""The filepaths to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        files = self.processed_file_names
        return [os.path.join(self.root, 'processed', self.split_dir, f) for f in files]

    def download(self):
        pass

    def __len__(self):
        return len(self.processed_file_names)

    def process(self):
        cnt_slices = 0
        progressBarPrefix = f'Generating samples for dataset G-ILSES2018 dataset type {self.dataset_type}'
        printProgressBar(0, len(self.indices), prefix=progressBarPrefix, length=50)
        # FIXME: raw path is incorrect path since in this case you cha ane trxta path called training.
        for case_id in self.raw_file_names:
            raw_path = os.path.join(self.root, 'raw', self.split_dir, case_id)
            patient_files = get_files_patient_path(raw_path)
            if len(self.modalities) > 1:
                ct_scan = np.stack(
                    [load_nifti(patient_files[mod][0], neurological_convension=True) for mod in self.modalities],
                    axis=-1)
            else:
                mod = self.modalities[0]
                ct_scan = load_nifti(patient_files[mod][0], neurological_convension=True)

            ct_scan = ct_scan.astype(np.float)
            ct_scan_norm = (ct_scan-ct_scan.min())/(ct_scan.max()-ct_scan.min())
            lesion_files = patient_files['LESION']
            lesion_mask = load_nifti(lesion_files[0], neurological_convension=True)
            # generates the graph inputs
            processed_num = len(lesion_mask)
            for i, case_index in enumerate(self.indices.get_by_case_id(case_id)):
                printProgressBar(cnt_slices + i, len(self.indices), prefix=progressBarPrefix, suffix=f'sample={case_index}', length=50)
                # Read data from `raw_path`.
                image = ct_scan_norm[i]
                mask = lesion_mask[i]
                if self.pre_transform is not None:
                    data = (image, mask)
                    data = self.pre_transform(data)
                else:
                    grid = grid_tensor((NORMALIZED_SHAPE['Y'], NORMALIZED_SHAPE['X']), connectivity=4)
                    num_elements = NORMALIZED_SHAPE['Y'] * NORMALIZED_SHAPE['X']
                    nodes_shape = num_elements if len(image.shape) <= 2 else (num_elements, -1)
                    grid.x = torch.tensor(image.reshape(nodes_shape)).float()
                    grid.y = torch.tensor([mask.reshape(num_elements)]).float()
                    data = grid

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                #torch.save(data, os.path.join(self.processed_dir, 'gendo_{:04d}.pt'.format(case_index)))
                torch.save(data, os.path.join(self.root, 'processed',
                    self.split_dir, 'gendo_{:04d}.pt'.format(case_index)))
            # update counter
            cnt_slices+=processed_num



    def get(self, idx):
        # compute offset
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))
        return data



class _ISLESFoldIndices:
    def __init__(self, cache_file, fold, dataset_type):
        self.cache_file = cache_file
        self.root = os.path.dirname(self.cache_file)
        self.cases_ids = None
        self.fold = fold
        self.dataset_type = dataset_type
        self._initialize_cases_id()
        if self._is_initialized():
            print('file exist loading indices from ' + self.cache_file)
            # self._load_indices()
        else:
            self._initialize()

    def _initialize(self):
        '''
        creates the indices for each case
        '''
        def get_offset():
            indices =  csv_to_dict(self.cache_file,',', item_col=0, key_col=1)
            return max([int(k) for k in indices.keys()])
        offset = get_offset() + 1 if os.path.exists(self.cache_file) else 0
        mode = 'a' if os.path.exists(self.cache_file) else 'w'
        with open(self.cache_file, mode, newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            for case_id in self.cases_ids:
                patient_files = get_files_patient_path(os.path.join(self.root, case_id))
                data = load_nifti(patient_files['LESION'][0], neurological_convension=True)
                num_elements= len(data)
                for i in range(num_elements):
                    csvwriter.writerow([case_id, i+offset])
                offset += num_elements
        self._load_indices()

    def _is_initialized(self):
        if not os.path.exists(self.cache_file):
            # the file doesn exist at all
            return False
        else:
            # checks if the file has cases id
            cases_id = self.get_cases()
            self._load_indices()
            for c in cases_id:
                if c not in self.indices.keys():
                    return False
            # after check all the cases_id returns True
            return True

    def _load_indices(self):
        '''
        Load the fold indices if the cache exists
        '''
        index_case_dict = csv_to_dict(self.cache_file, ',', key_col=1, item_col=0)
        self.indices = {}
        for case_id in self.get_cases():
            indices_case_id=[int(i) for i, c in index_case_dict.items() if c == case_id]
            if indices_case_id:
                self.indices[case_id] = indices_case_id
    def get_by_case_id(self, case_id):
        '''
        returns the cases ifor  agiven case id
        '''
        return self.indices[case_id]

    def __len__(self):
        length = 0
        for case_indices in self.indices.values():
            length += len(case_indices)
        return length

    def get_cases(self):
        return self.cases_ids

    def _initialize_cases_id(self):
        cases = []
        split_path = os.path.join(os.path.dirname(self.cache_file), 'split.txt')
        file_classes = csv_to_dict(split_path, ',', has_header=True, item_col=self.fold)
        for f, c in file_classes.items():
            if c == self.dataset_type:
                cases.append(f)
        self.cases_ids = cases
