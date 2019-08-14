import numpy as np
from scipy import ndimage

from .dataset import Datasets, Dataset
# CONSTANT WHERE TO FIND THE DATA
from config import VESSEL_DIR
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import Normalize
# import pyqtgraph as pg
import pandas as pd
from .download import maybe_download_and_extract
from imageio import imread


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


def erode_mask(mask):
    return ndimage.binary_erosion(mask, structure=np.ones((1, 7, 7)))


def read_dataset_mhd(data_dir, annotated_slices=True):
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
        lung_mask = erode_mask(lung_mask)
        ct_scan, origin, spacing = load_itk(os.path.join(data_dir, 'train', 'Scans', 'VESSEL12_{:02d}.mhd'.format(i)))
        ct_scan_masked = lung_mask*ct_scan
        ct_scan_masked.astype(np.float,copy=False)
        ct_scan_masked = (ct_scan_masked-ct_scan_masked.min())/(ct_scan_masked.max()-ct_scan_masked.min())

        vessel_mask, _ = load_vessel_mask_pre(ct_scan.shape, os.path.join(data_dir, 'train', 'Annotations', 'VESSEL12_{:02d}_OutputVolume.npy'.format(i)))
        vessel_mask = lung_mask*vessel_mask

        # this is legacy code commented because this is not industrial code XD
        # # alternatively, we may curate the 9 slices there fore we need to know which slices were annotated
        if annotated_slices:
            vessel_mask_annotations, z_slices = load_vessel_mask_csv(ct_scan.shape, os.path.join(data_dir, 'train', 'Annotations', 'VESSEL12_{:02d}_Annotations.csv'.format(i)))
            # ct_scan_masked = ct_scan_masked[z_slices]
            # vessel_mask = vessel_mask[z_slices]


        # plotting stuff...
        print('vessel_mask: ', vessel_mask.shape)
        print('origin: ', origin)
        print('spacing', spacing)
        print('z_slices', z_slices)
        im1 = ct_scan_masked/ct_scan_masked.max()
        im2 = vessel_mask/vessel_mask.max()
        # im1[im2==1]=2
        # pg.image(im1)
        # input('click to end')
        im2[vessel_mask_annotations==1]=-0.5
        vmax = np.abs(im2).max()
        vmin = -vmax
        cmap = plt.cm.RdYlBu

        for z_slice in z_slices:
            plt.figure()
            plt.imshow(im1[z_slice,:,:],cmap='gray')
            alphas = np.ones_like(im2[z_slice,:,:])
            alphas[im2[z_slice,:,:]==0]=0
            colors = Normalize(vmin, vmax, clip=True)(im2[z_slice,:,:])
            colors = cmap(colors)
            colors[..., -1] = alphas
            plt.imshow(colors)
            plt.show()
        images += [ct_scan_masked[i,:,:] for i in range(len(ct_scan_masked))]
        labels += [vessel_mask[i, :, :] for i in range(len(vessel_mask))]
    # TODO: split is hardcoded
    split = 0.2
    L = int(split*len(images))
    output['train']['images'], output['test']['images'] = np.stack(images[:-L], axis=0), np.stack(images[-L:], axis=0)
    output['train']['labels'], output['test']['labels'] = np.stack(labels[:-L], axis=0), np.stack(labels[-L:], axis=0)
    return output


def read_dataset_cured(data_dir):
    output = {'train': {'images': [], 'labels': []}, 'test': {'images': [], 'labels': []}}
    images = []
    labels = []
    raw_dir = os.path.join(data_dir,'Cured')
    maybe_download_and_extract('https://transfer.sh/3xwmz/cured_vessel12.zip',raw_dir)
    for i in range(9): # TODO: hardcode
        im = np.squeeze(imread(os.path.join(raw_dir,"lung{}.png".format(i)))).astype(np.float)
        lb = np.squeeze(imread(os.path.join(raw_dir, "mask{}.png".format(i)))).astype(np.float)
        mean = im.mean()
        std = im.std()
        im = (im-mean)/std
        lb = lb/lb.max()
        images.append(im)
        labels.append(lb)
    # TODO: again split is hardcoded
    split = 0.2
    L = int(split*len(images))
    output['train']['images'], output['test']['images'] = np.stack(images[:-L], axis=0), np.stack(images[-L:], axis=0)
    output['train']['labels'], output['test']['labels'] = np.stack(labels[:-L], axis=0), np.stack(labels[-L:], axis=0)

    return output


class VESSEL12(Datasets):
    def __init__(self,  data_dir=VESSEL_DIR, annotated_slices=False):
        if annotated_slices:
            vessel_data = read_dataset_cured(data_dir)
        else:
            vessel_data = read_dataset_mhd(data_dir)

        images = self._preprocess_images(vessel_data['train']['images'])
        labels = self._preprocess_labels(vessel_data['train']['labels'])
        train = Dataset(images, labels)

        images = self._preprocess_images(vessel_data['test']['images'])
        labels = self._preprocess_labels(vessel_data['test']['labels'])
        val = Dataset(images, labels)

        images = self._preprocess_images(vessel_data['test']['images'])
        labels = self._preprocess_labels(vessel_data['test']['labels'])
        test = Dataset(images, labels)

        super(VESSEL12, self).__init__(train, val, test)

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
        return 1

    def _preprocess_images(self, images):
        return np.expand_dims(images,axis=1)

    def _preprocess_labels(self, images):
        return images #np.expand_dims(images, axis=1)
        # threshold = 0.1
        # images = np.reshape(images, (-1, self.height, self.width,
        #                     self.num_channels))
        # masks = (images > threshold).astype(np.float)
        # return masks.squeeze()
