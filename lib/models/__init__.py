from .models import UNet, UnetSim

try:
    from .GFCN import GFCN
    __all__ = ['UnetSim', 'UNet', 'GFCN']
except:
    __all__ = ['UnetSim', 'UNet']