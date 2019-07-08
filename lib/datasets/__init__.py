from .mnist import MNIST
from .cifar_10 import Cifar10
from .pascal_voc import PascalVOC
from .queue import PreprocessQueue
from .m2nist import M2NIST
from .vessel12 import VESSEL12
from .gvessel12 import GVESSEL12
try:
    from .gmnist import GMNIST
except ImportError:
    print('Warning: Error while importing lib.GMINST module. Pytorch Geometric not installed')

__all__ = ['MNIST', 'Cifar10', 'PascalVOC', 'PreprocessQueue', 'M2NIST', 'GMNIST', 'VESSEL12', 'GVESSEL12']
