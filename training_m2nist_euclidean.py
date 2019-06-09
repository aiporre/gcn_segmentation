from lib.models import UNet
from lib.datasets import M2NIST
from lib.process import Trainer
dataset = M2NIST()
model = UNet(n_channels=1, n_classes=1)
trainer = Trainer(model=model,dataset=dataset, batch_size=32)

for _ in range(10):
    loss = trainer.train_epoch()
    print('loss',loss)
