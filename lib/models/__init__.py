from .models import UNet, UnetSim, FCN

try:
    from .GFCN import GFCN, GFCNA
    __all__ = ['UnetSim', 'UNet', 'FCN', 'GFCN', 'GFCNA']
except:
    __all__ = ['UnetSim', 'UNet', 'FCN']