import argparse

from lib.models import GFCN, GFCNA
from lib.datasets import GVESSEL12
from lib.process import Trainer, Evaluator
import matplotlib.pyplot as plt
import torch
from config import VESSEL_DIR
from lib.utils import savefigs
import numpy as np



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
    parser.add_argument("-d", "--vesseldir", type=str, default=VESSEL_DIR,
                        help="parameter gamam of the gaussians")
    parser.add_argument("-f", "--figdir", type=str, default='./fig',
                        help="path to save figs")
    parser.add_argument("-b", "--batch", type=int, default=2,
                        help="batch size of trainer and evaluator")
    return parser.parse_args()

# CONSTANST

args = process_command_line()
EPOCHS = args.epochs
MODEL_PATH = './GFCN-vessel12-full_slices.pth'
EPOCHS = args.epochs
BATCH = args.batch
dataset = GVESSEL12(data_dir=args.vesseldir)
model = GFCNA()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

trainer = Trainer(model=model,dataset=dataset, batch_size=BATCH,to_tensor=False, device=device)
trainer.load_model(model, MODEL_PATH)
evaluator = Evaluator(dataset=dataset, batch_size=BATCH, to_tensor=False, device=device)

def train(lr=0.001, progress_bar=False):
    loss_all =[]
    for _ in range(EPOCHS):
        loss = trainer.train_epoch(lr=lr, progress_bar=progress_bar)
        print('loss epoch', np.array(loss).mean())
        loss_all += loss
        with torch.no_grad():
            score = evaluator.DCM(model, progress_bar=progress_bar)
            print('DCM score:', score)
    plt.plot(loss_all)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('loss history epochs')
    print('end of training')
    trainer.save_model(MODEL_PATH)

def eval(lr=0.001, progress_bar=False, fig_dir='./figs'):
    # print('DCM factor: ' , evaluator.DCM(model, progress_bar=progress_bar))
    print('plotting one prediction')
    fig = evaluator.plot_prediction(model=model)
    savefigs(fig_name='gfcn_e{}_lr{}_annotated_slices'.format(EPOCHS, lr),fig_dir=fig_dir, fig=fig)
    plt.show()

train(lr=args.lr, progress_bar=args.progressbar)
eval(lr=args.lr, progress_bar=args.progressbar, fig_dir=args.figdir)
