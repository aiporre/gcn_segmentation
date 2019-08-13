import numpy as np
from scipy.ndimage import gaussian_filter

from .dataset import Datasets, Dataset
# CONSTANT WHERE TO FIND THE DATA
from config import SVESSEL_DIR as VESSEL_DIR
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


def read_dataset_mhd(data_dir):
    '''
        Reads the directory and conforms the structure of generic datasets:
        {'train': {'images': list of images, 'labels': list of labels}
         'test': {'images': list of images, 'labels': list of labels}}
    '''
    output = {'train': {'images':[], 'labels':[]} , 'test': {'images':[], 'labels':[]}}
    images = []
    labels = []
    for i in range(1,11):
        for j in range(8,13):
            # reading the ct-scan masked with the lungs
            ct_scan, origin, spacing = load_itk(os.path.join(data_dir, 'Group{}'.format(i), 'data{}'.format(j), 'testVascuSynth{}_101_101_101_uchar.mhd'.format(j)))
            ct_scan = ct_scan/ct_scan.max()# normalization

            vessel_mask =  ct_scan>0
            vessel_mask = vessel_mask.astype(np.float,copy=False)
            ct_scan_real = gaussian_filter(ct_scan, sigma=0.5) + np.random.normal(0, 0.05, ct_scan.shape)
            ct_scan_real = (ct_scan_real-ct_scan_real.min())/(ct_scan_real.max()-ct_scan_real.min())# normalization


            # z_slices = [10,50,70]
            # # plotting stuff...
            # print('vessel_mask: ', vessel_mask.shape)
            # print('origin: ', origin)
            # print('spacing', spacing)
            # im1 = ct_scan/ct_scan.max()
            # im2 = vessel_mask/vessel_mask.max()
            # im3 = ct_scan_real/ct_scan_real.max()
            # # im1[im2==1]=2
            # # pg.image(im1)
            # # input('click to end')
            #
            # for z_slice in z_slices:
            #     fig = plt.figure()
            #     ax1 = fig.add_subplot(131)  # left side
            #     ax2 = fig.add_subplot(132)  # right side
            #     ax3 = fig.add_subplot(133)  # right side
            #     ax1.imshow(im1[z_slice,:,:],cmap='gray')
            #     ax2.imshow(im2[z_slice, :, :], cmap='gray')
            #     ax3.imshow(im3[z_slice, :, :], cmap='gray')
            #     ax1.axis('off')
            #     ax2.axis('off')
            #     ax3.axis('off')
            #
            #     plt.show()
            images += [ct_scan_real[i,:,:] for i in range(len(ct_scan_real))]
            labels += [vessel_mask[i, :, :] for i in range(len(vessel_mask))]
    # TODO: split is hardcoded
    split = 0.2
    L = int(split*len(images))
    output['train']['images'], output['test']['images'] = np.stack(images[:-L], axis=0), np.stack(images[-L:], axis=0)
    output['train']['labels'], output['test']['labels'] = np.stack(labels[:-L], axis=0), np.stack(labels[-L:], axis=0)
    return output

class SVESSEL(Datasets):
    def __init__(self,  data_dir=VESSEL_DIR):
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

        super(SVESSEL, self).__init__(train, val, test)

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
