from lib.models import UNet
from lib.datasets import GMNIST
from lib.process import Trainer, Evaluator
import matplotlib.pyplot as plt
import torch
# CONSTANST
MODEL_PATH = './u-net-mnist.pth'
EPOCHS = 1

dataset = GMNIST()




# model = UNet(n_channels=1, n_classes=1)
# trainer = Trainer(model=model,dataset=dataset, batch_size=64)
# trainer.load_model(model, MODEL_PATH)
# evaluator = Evaluator(dataset=dataset)
#
# def train():
#     for _ in range(EPOCHS):
#         loss = trainer.train_epoch()
#         print('loss',loss)
#         with torch.no_grad():
#             score = evaluator.DCM(model=model)
#             print('DCM score:', score)
#     print('end of training')
#     trainer.save_model(MODEL_PATH)
#
# def eval():
#     model.eval()
#     print('DCM factor: ' , evaluator.DCM(model))
#     print('plotting one prediction')
#     fig = evaluator.plot_prediction(model=model)
#     plt.show()
#
# train()
# eval()