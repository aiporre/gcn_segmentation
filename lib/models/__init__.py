from .models import UNet, FCN

try:
    from .GFCN import GFCN, GFCNA, GFCNB, GFCNC, GFCND, GFCNE, GFCNG
    from .pointnet import PointNet
    __all__ = [ 'UNet', 'FCN', 'GFCN', 'GFCNA', 'GFCNB', 'GFCNC', 'GFCND', 'GFCNE', 'GFCNG']
except:
    __all__ = [ 'UNet', 'FCN', 'PointNet']