from .models import UNet, FCN

try:
    from .GFCN import GFCN, GFCNA, GFCNB, GFCNC, GFCND, GFCNE, GFCNF, GFCNG
    from .pointnet import PointNet
    __all__ = [ 'UNet', 'FCN', 'GFCN', 'GFCNA', 'GFCNB', 'GFCNC', 'GFCND', 'GFCNE', 'GFCNF', 'GFCNG']
except:
    __all__ = [ 'UNet', 'FCN', 'PointNet']