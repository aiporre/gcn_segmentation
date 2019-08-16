import argparse

try:
    from lib.models import GFCN, GFCNA, GFCNC, GFCNB, PointNet
    from lib.datasets import GMNIST, GSVESSEL, GVESSEL12
except:
    print('No module torch geometric')
from lib.models import UNet
from lib.datasets import MNIST, VESSEL12, SVESSEL, Crop


from lib.process import Trainer, Evaluator, DCS
import matplotlib.pyplot as plt
import torch
from torch import nn

from config import VESSEL_DIR, SVESSEL_DIR
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
    parser.add_argument("-vd", "--vesseldir", type=str, default=VESSEL_DIR,
                        help=" Vessel12 dataset dir")
    parser.add_argument("-sd", "--svesseldir", type=str, default=VESSEL_DIR,
                        help="syntetic vessel dataset dir")
    parser.add_argument("-f", "--figdir", type=str, default='./fig',
                        help="path to save figs")
    parser.add_argument("-b", "--batch", type=int, default=2,
                        help="batch size of trainer and evaluator")
    parser.add_argument("-s", "--dataset", type=str, default='MNIST',
                        help="dataset to be used")
    parser.add_argument("-n", "--net", type=str, default='GFCN',
                            help="network to be used")
    parser.add_argument("-p", "--pre-transform", type=bool, default=False,
                        help="use a pretransfrom to the dataset")
    parser.add_argument("-c", "--criterion", type=str, default='BCE',
                        help="criterion: BCE or DCS or BCElogistic or DCSsigmoid")
    return parser.parse_args()

# CONSTANST

args = process_command_line()
EPOCHS = args.epochs
MODEL_PATH = './{}-ds{}.pth'.format(args.net, args.dataset)
EPOCHS = args.epochs
BATCH = args.batch

if args.pre_transform:
    pre_transform = Crop(30,150,256,256)
else:
    pre_transform = None
if args.dataset == 'MNIST':
    dataset = MNIST()
elif args.dataset == 'GMNIST':
    dataset = GMNIST()
elif args.dataset == 'VESSEL12':
    dataset = VESSEL12(data_dir=args.vesseldir, pre_transform=pre_transform)
elif args.dataset == 'GVESSEL12':
    dataset = GVESSEL12(data_dir=args.vesseldir, pre_transform=pre_transform)
else:
    dataset = MNIST()


if args.net=='GFCN':
    model = GFCN()
elif args.net == 'GFCNA':
    model = GFCNA()
elif args.net == 'GFCNB':
    model = GFCNB()
elif args.net == 'GFCNC':
    model = GFCNC()
elif args.net=='PointNet':
    model = PointNet()
elif args.net == 'UNet':
    model = UNet(n_channels=1, n_classes=1)
else:
    model = GFCNA()

if args.criterion == 'BCE':
    criterion = nn.BCELoss()
    sigmoid=True
elif args.criterion == 'BCElogistic':
    criterion = nn.BCEWithLogitsLoss()
    sigmoid = False
elif args.criterion == 'DCS':
    criterion = DCS()
    sigmoid = True
elif args.criterion == 'DCSsigmoid':
    criterion = DCS(pre_sigmoid=True)
    sigmoid = False
else:
    criterion = nn.BCELoss()
    sigmoid = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
if args.dataset[0] == 'G':
    trainer = Trainer(model=model,dataset=dataset, batch_size=BATCH,to_tensor=False, device=device, criterion=criterion)
    evaluator = Evaluator(dataset=dataset, batch_size=BATCH, to_tensor=False, device=device, sigmoid=sigmoid)
    trainer.load_model(model, MODEL_PATH)
else:
    trainer = Trainer(model=model, dataset=dataset, batch_size=BATCH, device=device, criterion=criterion)
    evaluator = Evaluator(dataset=dataset, batch_size=BATCH, device=device, sigmoid=sigmoid)
    trainer.load_model(model, MODEL_PATH)


def train(lr=0.001, progress_bar=False, fig_dir='./figs',prefix='NET'):
    loss_all = []
    loss_epoch = []
    DCS = []
    A = []
    P = []
    R = []
    for e in range(EPOCHS):
        model.train()
        loss = trainer.train_epoch(lr=lr, progress_bar=progress_bar)
        mean_loss = np.array(loss).mean()
        loss_epoch.append(mean_loss)
        print('EPOCH ', e, 'loss epoch', mean_loss)
        loss_all += loss
        with torch.no_grad():
            model.eval()
            DCS.append(evaluator.DCM(model, progress_bar=progress_bar))
            a, p, r  = evaluator.bin_scores(model, progress_bar=progress_bar)
            P.append(p)
            A.append(a)
            R.append(r)
            print('DCS score:', DCS[-1], 'accuracy ', a, 'precision', p, 'recall', r )
    fig = plt.figure(figsize=(10,10))
    loss_all = np.array(loss_all)
    measurements = np.array([DCS,P,A,R])
    np.save('{}_e{}_lr{}_ds{}_lossall'.format(prefix, EPOCHS, lr, args.dataset), measurements)
    np.save('{}_e{}_lr{}_ds{}_measurements'.format(prefix, EPOCHS, lr, args.dataset),measurements)

    plt.subplot(3,1,1)
    plt.plot(loss_all)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('loss history')
    plt.subplot(3,1,2)
    plt.plot(loss_epoch)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('loss history')
    plt.subplot(3,1,3)
    plt.plot(DCS)
    plt.plot(P)
    plt.plot(A)
    plt.plot(R)
    plt.xlabel('epochs')
    plt.ylabel('metrics')
    plt.title('metrics')
    savefigs(fig_name='{}_e{}_lr{}_ds{}_loss_history'.format(prefix, EPOCHS, lr, args.dataset), fig_dir=fig_dir, fig=fig)
    print('end of training')
    trainer.save_model(MODEL_PATH)

def eval(lr=0.001, progress_bar=False, fig_dir='./figs',prefix='NET'):
    model.eval()
    print('DCM factor: ' , evaluator.DCM(model, progress_bar=progress_bar))
    print('plotting one prediction')
    fig = evaluator.plot_prediction(model=model)
    savefigs(fig_name='{}_e{}_lr{}_ds{}_performance'.format(prefix,EPOCHS, lr, args.dataset),fig_dir=fig_dir, fig=fig)
    plt.show()

train(lr=args.lr, progress_bar=args.progressbar, fig_dir=args.figdir, prefix=args.net)
eval(lr=args.lr, progress_bar=args.progressbar, fig_dir=args.figdir, prefix=args.net)