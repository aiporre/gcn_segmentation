from lib.models import UNet
from lib.datasets import VESSEL12
from lib.process import Trainer, Evaluator
import matplotlib.pyplot as plt
import torch
# CONSTANST
MODEL_PATH = './u-net-vessel12.pth'
EPOCHS = 1

dataset = VESSEL12()
model = UNet(n_channels=1, n_classes=1)
trainer = Trainer(model=model,dataset=dataset, batch_size=4)
trainer.load_model(model, MODEL_PATH)
evaluator = Evaluator(dataset=dataset)

def train():
    for _ in range(EPOCHS):
        loss = trainer.train_epoch()
        print('loss',loss)
        with torch.no_grad():
            score = evaluator.DCM()
            print('DCM score:', score)
    print('end of training')
    trainer.save_model(MODEL_PATH)

def eval():
    # print('DCM factor: ' , evaluator.DCM(model))
    print('plotting one prediction')
    fig = evaluator.plot_prediction(model=model)
    plt.show()

train()
eval()