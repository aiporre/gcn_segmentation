import csv
import re

import numpy as np
from scipy import ndimage

# CONSTANT WHERE TO FIND THE DATA
from config import ISLES2018_DIR
import os
from .dataset import Datasets, Dataset, EuclideanDataset
import torch
import numpy as np
from lib.process.progress_bar import printProgressBar
from lib.utils.csv import csv_to_dict
import nibabel as nib
from pathlib import Path

NORMALIZED_SHAPE = {'Z': None, 'Y': 256, 'X': 256}


def get_modalities(arg_mod):
    modalities = {"CTN": "CTN", "TMAX": "CTP-TMAX", "CBF": "CTP-CBF", "CBV": "CTP-CBV", "MTT": "CTP-MTT"}
    output_modalities = [modalities[mod] for mod in arg_mod]
    return output_modalities


def isles2018_reshape(x, channels=None, channel=None):
    '''
    Transform used by plotting functions
    '''
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    N = x.size
    nn = NORMALIZED_SHAPE['Y'] * NORMALIZED_SHAPE['X']
    # channels are deduced from N // nn if input is None
    N_channels = N // nn if channels is None else channels
    if N_channels == 1:
        isles_shape = (NORMALIZED_SHAPE['Y'], NORMALIZED_SHAPE['X'])
    else:
        isles_shape = (NORMALIZED_SHAPE['Y'], NORMALIZED_SHAPE['X'], N_channels)
    x = np.reshape(x, isles_shape)
    # selects one channel in specificed by channel input
    if N_channels > 1 and channel is not None:
        x = x[..., channel]
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
        data = np.transpose(data, (2, 1, 0))
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


class ISLES2018(Datasets):
    def __init__(self, data_dir=ISLES2018_DIR,
                 batch_size=32,
                 test_rate=0.2,
                 annotated_slices=False,
                 pre_transform=None,
                 fold=1,
                 modalities=("CTN", "CTP-TMAX", "CTP-CBF", "CTP-CBV", "CTP-MTT"),
                 useful=False):
        self._num_channels = len(modalities)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_rate = test_rate
        train_images = _ISLES2018(self.data_dir, dataset_type='train', transform=None,
                                    pre_transform=pre_transform, fold=fold, modalities=modalities, useful=useful)
        val_images = _ISLES2018(self.data_dir, dataset_type='val', transform=None,
                                  pre_transform=pre_transform, fold=fold, modalities=modalities, useful=useful)
        test_images = _ISLES2018(self.data_dir, dataset_type='test', transform=None,
                                   pre_transform=pre_transform, fold=fold, modalities=modalities, useful=useful)

        train = EuclideanDataset(dataset=train_images)
        val = EuclideanDataset(dataset=val_images)
        test = EuclideanDataset(dataset=test_images)
        super(ISLES2018, self).__init__(train=train, test=test, val=val)

    @property
    def classes(self):
        return ['foreground', 'background']

    @property
    def width(self):
        return 512

    @property
    def height(self):
        return 512

    @property
    def num_channels(self):
        return self._num_classes

def makedirs(path):
    try:
        import os
        os.makedirs(os.path.expanduser(os.path.normpath(path)))
    except OSError as e:
        print("Could not create directory %s: %s" % (path,e))
        raise e

class _ISLES2018(object):

    def __init__(self,
                 root,
                 transform=None,
                 dataset_type='train',
                 pre_transform=None,
                 pre_filter=None,
                 fold=1,
                 split_dir="TRAINING",
                 modalities=("CTN", "CTP-TMAX", "CTP-CBF", "CTP-CBV", "CTP-MTT"),
                 useful=False):
        self.labels = True
        self.root = root
        self.test_rate = 1
        self.fold = fold
        self.dataset_type = dataset_type
        self.split_dir = split_dir
        self.modalities = modalities
        # creates the processed dir
        raw_dir = os.path.join(self.root, 'raw', self.split_dir)
        processed_dir = os.path.join(self.root, 'processed_euclid', self.split_dir)
        makedirs(processed_dir)
        # procesed mapping is csv file that tell which proccesd files match with which case_XY
        # e.g. gilses_1000 --> case_10
        self.indices = _ISLESFoldIndices(cache_file=os.path.join(root, 'raw', self.split_dir, 'processed_mapping.txt'),
                                         fold=fold, dataset_type=dataset_type)
        self.useful = useful # flag that activates getting only the relevant samples
        # with masks bigger that 1000 pixels which is equivalent to nearby 1%
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir

    @property
    def raw_file_names(self):
        files = self.indices.get_cases()
        return files

    @property
    def processed_file_names(self):
        processed_files = []
        for case_id in self.raw_file_names:
            processed_indices = self.indices.get_by_case_id(case_id, useful=self.useful)
            _processed_files = [self._get_proc_file(i) for i in processed_indices]
            processed_files.extend(_processed_files)
        return processed_files

    @property
    def raw_paths(self):
        r"""The filepaths to find in order to skip the download."""
        files = self.raw_file_names
        return [os.path.join( self.raw_dir, f) for f in files]

    @property
    def processed_paths(self):
        r"""The filepaths to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        files = self.processed_file_names
        return [os.path.join(self.processed_dir, f) for f in files]


    def __len__(self):
        return len(self.processed_file_names)

    def process(self):
        cnt_slices = 0
        progressBarPrefix = f'Generating samples for dataset ILSES2018 dataset type {self.dataset_type}'
        printProgressBar(0, len(self.indices), prefix=progressBarPrefix, length=50)

        for case_id in self.raw_file_names:
            raw_path = os.path.join(self.root, 'raw', self.split_dir, case_id)
            patient_files = get_files_patient_path(raw_path)

            def norm(x):
                return (x - x.min()) / (x.max() - x.min())
            # default is the idem function
            skull_strippen = lambda x: x
            # if the CT native is selected then use stry based od the non-zero sum values of the CTP-parameters
            if "CTN" in self.modalities:
                brain_mask = 1
                for m in ["CTP-TMAX", "CTP-CBF", "CTP-CBV", "CTP-MTT"]:
                    brain_mask += (norm(load_nifti(patient_files[m][0], neurological_convension=True)) > 0).astype(np.float)
                skull_strippen = lambda x: x * brain_mask
            if len(self.modalities) > 1:
                ct_scan = []
                for mod in self.modalities:
                    x = load_nifti(patient_files[mod][0], neurological_convension=True).astype(np.float)
                    x = norm(x)
                    if mod == "CTN":
                        x = skull_strippen(x)
                    ct_scan.append(x)
                ct_scan = np.stack(ct_scan, axis=-1)
            else:
                mod = self.modalities[0]
                ct_scan = norm(load_nifti(patient_files[mod][0], neurological_convension=True).astype(np.float))
                if mod == "CTN":
                    ct_scan = skull_strippen(ct_scan)
            lesion_files = patient_files['LESION']
            lesion_mask = load_nifti(lesion_files[0], neurological_convension=True)
            # generates the 2D euclidean inputs to train the euclidean models
            processed_num = len(lesion_mask)
            for i, processed_index in enumerate(self.indices.get_by_case_id(case_id)):
                printProgressBar(cnt_slices + i, len(self.indices), prefix=progressBarPrefix,
                                 suffix=f'sample={processed_index}', length=50)
                # Read data from `raw_path`.
                image = ct_scan[i]
                mask = lesion_mask[i]
                num_elements = NORMALIZED_SHAPE['Y'] * NORMALIZED_SHAPE['X']
                nodes_shape = num_elements if len(image.shape) <= 2 else (num_elements, -1)
                x = torch.tensor(image).float()
                y = torch.tensor(mask).float()
                # generates a tuple tensor containting first the images (n-channels) and seconds element is the mask
                data = torch.stack([x,y], dim=0)
                torch.save(data, os.path.join(self.processed_dir, self._get_proc_file(processed_index)))
            # update counter
            cnt_slices += processed_num

    def get_all_cases_id(self):
        return self.raw_file_names

    def get_by_case_id(self, case_id, useful=None):
        if useful is None:
            # override with the one of the dataset
            useful = self.useful
        case_id_indices = self.indices.get_by_case_id(case_id, useful=useful)
        case_id_processed_file_names = [self._get_proc_file(case_id_index) for case_id_index in case_id_indices]
        self_useful = self.useful
        for c_id_fname in case_id_processed_file_names:
            self.useful = useful
            c_id_idx = self.processed_file_names.index(c_id_fname)
            data = self.__getitem__(c_id_idx)
            self.useful = self_useful
            yield data

    def get_indices_by_case_id(self, case_id, useful=None, relative_dataset=False):
        if useful is None:
            # override with the one of the dataset
            useful = self.useful
        if not relative_dataset:
            # returns indices for a given case id absolute, as specified in the file processed_file.txt
            return self.indices.get_by_case_id(case_id, useful=useful)
        else:
            indices_absolute = self.indices.get_by_case_id(case_id, useful=useful)
            processed_file_names = [self._get_proc_file(case_id_index) for case_id_index in indices_absolute]
            indices_relative_to_dataset = []
            self_useful = self.useful
            self.useful = useful
            for pf in processed_file_names:
                indices_relative_to_dataset.append(self.processed_file_names.index(pf))
            self.useful = self_useful
            return indices_relative_to_dataset

    def get_case_id(self, index):
        # turn off flag to access all indices
        self_useful = self.useful
        self.useful = False
        # Index correspond to the relative to the dataset.
        # For example, index 0 is the index absolute 1280 in the test dataset
        index_absolute = int(re.search(r'\d+', self.processed_file_names[index]).group())
        case_id = self.indices.get_case_id(index_absolute)
        # turn on it back
        self.useful = self_useful
        return case_id

    def get(self, idx):
        # compute offset
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))
        return data

    def _get_proc_file(self, processed_index):
        return 'endo_{:04d}.pt'.format(processed_index)

    def make_labels(self):
        # TODO: I dont know if I should make this an instance of the class lib.dataset.Dataset so it get the correct
        # TODO: this requires to modified the parent class so it can be initializied without the list of files
        # TODO: Make labels isnt the most appropriate method as it creates a new instace with the same object and changes behavrio, this is probably to complicated, the data is already there
        # TODO: this calss will have probelms iwththe get get indices by case where the method get is by passed you returun x and not y, or viceversa.
        self._dlabel_dataset = True
        return self

    def __getitem__(self, item):
        data = self.get(item)
        if self.labels:
            return data[1]
        else:
            return data[0]





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
            indices = csv_to_dict(self.cache_file, ',', item_col=0, key_col=1)
            return max([int(k) for k in indices.keys()])

        offset = get_offset() + 1 if os.path.exists(self.cache_file) else 0
        mode = 'a' if os.path.exists(self.cache_file) else 'w'
        with open(self.cache_file, mode, newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            for case_id in self.cases_ids:
                patient_files = get_files_patient_path(os.path.join(self.root, case_id))
                data = load_nifti(patient_files['LESION'][0], neurological_convension=True)
                num_elements = len(data)
                useful_scans = list(data.sum(axis=(1, 2)) > 500)
                for i in range(num_elements):
                    csvwriter.writerow([case_id, i + offset, str(useful_scans[i])])
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
        index_useful_dict = csv_to_dict(self.cache_file, ',', key_col=1, item_col=2)
        self.indices = {}
        self.useful_indices = {}
        self.indices_backward = {}
        for case_id in self.get_cases():
            indices_case_id = [int(i) for i, c in index_case_dict.items() if c == case_id]
            useful_case_id = [index_useful_dict[str(i)] == 'True' for i in indices_case_id]
            sample_case_id = { i: c for i, c in index_case_dict.items() if c == case_id}
            if indices_case_id:
                self.indices[case_id] = indices_case_id
                self.useful_indices[case_id] = useful_case_id
                self.indices_backward = dict(self.indices_backward, **sample_case_id)

    def get_by_case_id(self, case_id, useful=False):
        '''
        returns the cases ifor  agiven case id
        '''
        if not useful:
            return self.indices[case_id]
        else:
            indices = self.indices[case_id]
            useful_indices = self.useful_indices[case_id]
            return [idx for idx, useful in zip(indices, useful_indices) if useful]

    def get_case_id(self, index):
        # get case id from index. eg. 10 ==> case_id = 1
        # only indices in the type of dataset are keys in this dictionary
        if isinstance(index, int):
            index = str(index)
        return self.indices_backward[index]

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
