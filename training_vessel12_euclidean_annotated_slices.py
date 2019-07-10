from lib.models import UNet
from lib.datasets import VESSEL12
from lib.process import Trainer, Evaluator
from lib.utils import savefigs
import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse


# CONSTANST
MODEL_PATH = './u-net-vessel12_annotated_slices.pth'
EPOCHS = 200

dataset = VESSEL12('./data/vessel12/', annotated_slices=True)
model = UNet(n_channels=1, n_classes=1)
trainer = Trainer(model=model,dataset=dataset, batch_size=4)
trainer.load_model(model, MODEL_PATH)
evaluator = Evaluator(dataset=dataset)

def train(lr = 0.001, progress_bar=False):
    loss_all = []
    for _ in range(EPOCHS):
        loss = trainer.train_epoch(lr=lr, progress_bar=progress_bar)
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

def eval(progress_bar = False, figs_dir='./figs'):
    # print('DCM factor: ' , evaluator.DCM(model,progress_bar=progress_bar))
    print('plotting one prediction')
    fig = evaluator.plot_prediction(model=model)
    savefigs(fig_name='unet_e{}_lr{}_annotatedslices.png', fig=fig, fig_dir=figs_dir)
    plt.show()



def process_command_line():
    """Parse the command line arguments.
    """
    parser = argparse.ArgumentParser(description="Machine Learning exercise 5.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--progressbar", type=bool, default=False,
                        help="progress bar continuous")
    parser.add_argument("-lr", "--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("-g", "--epochs", type=int, default=10,
                        help="parameter gamam of the gaussians")

    parser.add_argument("-f", "--figsdir", type=str, default='./fig',
                        help="path to save figs")

    return parser.parse_args()



if __name__ == '__main__':
    args = process_command_line()
    EPOCHS = args.epochs
    train(lr=args.lr, progress_bar=args.progressbar)
    eval(progress_bar=args.progressbar, figs_dir=args.figsdir)


