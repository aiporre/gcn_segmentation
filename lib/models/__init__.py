from .models import UNet, UnetSim

try:
    from .GFCN import GFCN, GFCNA
    __all__ = ['UnetSim', 'UNet', 'GFCN', 'GFCNA']
except:
    __all__ = ['UnetSim', 'UNet']