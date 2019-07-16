import argparse

from lib.models import FCN
from lib.datasets import VESSEL12
from lib.process import Trainer, Evaluator
import matplotlib.pyplot as plt
import torch
import numpy as np
from config import VESSEL_DIR
from lib.utils import savefigs


def process_command_line():
    """Parse the command line arguments.
    """
    parser = argparse.ArgumentParser(description="Training Euclidean all",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--progressbar", type=bool, default=False,
                        help="progress bar continuous")
    parser.add_argument("-lr", "--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("-g", "--epochs", type=int, default=10,
                        help=" number of epochs")
    parser.add_argument("-d", "--vesseldir", type=str, default=VESSEL_DIR,
                        help="directory of vessels")
    parser.add_argument("-f", "--figsdir", type=str, default='./figs',
                        help="path to save figs")
    parser.add_argument("-b", "--batch", type=int, default=2,
                        help="batch size of trainer and evaluator")
    return parser.parse_args()

# CONSTANST

args = process_command_line()

MODEL_PATH = './FCN-vessel12_full_slices.pth'
EPOCHS = 1
EPOCHS = args.epochs
BATCH = args.batch
dataset = VESSEL12(data_dir=args.vesseldir)

model = FCN(n_channels=1, n_classes=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

trainer = Trainer(model=model,dataset=dataset, batch_size=BATCH, device=device)
trainer.load_model(model, MODEL_PATH)
evaluator = Evaluator(dataset=dataset, batch_size=BATCH, device=device)

def train(lr = 0.001, progress_bar=False):
    loss_all = []
    for _ in range(EPOCHS):
        loss = trainer.train_epoch(lr=lr, progress_bar=progress_bar)
        print('loss epoch',np.array(loss).mean())
        loss_all +=loss
        with torch.no_grad():
            score = evaluator.DCM(model, progress_bar=progress_bar)
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
    savefigs(fig_name='FCN_e{}_lr{}_full_slices'.format(EPOCHS,0.001), fig=fig, fig_dir=figs_dir)
    plt.show()


train(lr=args.lr, progress_bar=args.progressbar)
eval(progress_bar=args.progressbar, figs_dir=args.figsdir)
