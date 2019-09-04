from .mnist import MNIST
from .cifar_10 import Cifar10
from .pascal_voc import PascalVOC
from .queue import PreprocessQueue
from .m2nist import M2NIST
from .vessel12 import VESSEL12
from .svessel import SVESSEL
try:
    from .transforms import Crop, CropVessel12
except ImportError as e:
    print('Warning: Error while importing lib.Crop, lib.CropVessel12 module. Pytorch Geometric not installed, ', str(e))

try:
    from .gvessel12 import GVESSEL12
except ImportError as e:
    print('Warning: Error while importing lib.GVESSEL12 module. Pytorch Geometric not installed, ', str(e))

try:
    from .gsvessel import GSVESSEL
except ImportError as e:
    print('Warning: Error while importing lib.GSVESSEL module. Pytorch Geometric not installed, ', str(e))

try:
    from .gmnist import GMNIST
except ImportError as e:
    print('Warning: Error while importing lib.GMINST module. Pytorch Geometric not installed, ', str(e))

__all__ = ['MNIST', 'Cifar10', 'PascalVOC', 'PreprocessQueue', 'M2NIST', 'GMNIST', 'VESSEL12', 'GVESSEL12', 'Crop','CropVessel12', 'GSVESSEL', 'SVESSEL']
