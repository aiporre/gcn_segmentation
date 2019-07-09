from lib.models import UNet
from lib.datasets import VESSEL12
from lib.process import Trainer, Evaluator
import matplotlib.pyplot as plt
import torch
import numpy as np
# CONSTANST
MODEL_PATH = './u-net-vessel12_annotated_slices.pth'
EPOCHS = 100

dataset = VESSEL12('./data/vessel12/', annotated_slices=True)
model = UNet(n_channels=1, n_classes=1)
trainer = Trainer(model=model,dataset=dataset, batch_size=4)
trainer.load_model(model, MODEL_PATH)
evaluator = Evaluator(dataset=dataset)

def train():
    loss_all = []
    for _ in range(EPOCHS):
        loss = trainer.train_epoch(lr=0.1,progress_bar=False)
        print('loss epoch',np.array(loss).mean())
        loss_all +=loss
        with torch.no_grad():
            score = evaluator.DCM(model)
            print('DCM score:', score)
    plt.plot(loss_all)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('loss history epochs')

    print('end of training')
    trainer.save_model(MODEL_PATH)

def eval():
    # print('DCM factor: ' , evaluator.DCM(model))
    print('plotting one prediction')
    fig = evaluator.plot_prediction(model=model)
    plt.show()

# train()
eval()