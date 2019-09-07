import argparse

try:
    from lib.datasets import GMNIST
except Exception as e:
    print('Warning: No module torch geometric. Failed to import GMISNT, Exception: ', str(e))

try:
    from lib.datasets import GSVESSEL
except Exception as e:
    print('Warning: No module torch geometric. Failed to import GSVESSEL, Exception: ', str(e))

try:
    from lib.datasets import GVESSEL12
except Exception as e:
    print('Warning: No module torch geometric. Failed to import GVESSEL12, Exception: ', str(e))

try:
    from lib.models import GFCN, GFCNA, GFCNC, GFCNB, PointNet, GFCND
except Exception as e:
    print('Warning: No module torch geometric. Failed to import models, Exception: ', str(e))
    
try:
    from dvn import FCN as DeepVessel
except Exception as e:
    print('Warning: No module dvn. Failed to import deep vessel models, Exception: ', str(e))
    
from lib.models import UNet, FCN
from lib.datasets import MNIST, VESSEL12, SVESSEL, Crop, CropVessel12


from lib.process import Trainer, Evaluator, DCS , KEvaluator, KTrainer
import matplotlib.pyplot as plt
import torch
from torch import nn

from config import VESSEL_DIR, SVESSEL_DIR
from lib.utils import savefigs, Timer
import numpy as np
from keras import backend as K



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def process_command_line():
    """Parse the command line arguments.
    """


    parser = argparse.ArgumentParser(description="Machine Learning Training: :)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--progressbar", type=str2bool, default=False,
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
                        help="dataset to be used. Options: (G)MNIST, (G)VESSEL12, (G)SVESSEL")
    parser.add_argument("-n", "--net", type=str, default='GFCN',
                            help="network to be used. ...." )
    parser.add_argument("-p", "--pre-transform", type=str2bool, default=False,
                        help="use a pretransfrom to the dataset")
    parser.add_argument("-z", "--background", type=str2bool, default=True,
                        help="use a background in the MNIST dataset.")
    parser.add_argument("-c", "--criterion", type=str, default='BCE',
                        help="criterion: BCE or DCS or BCElogistic or DCSsigmoid")
    parser.add_argument("-u", "--upload", type=str2bool, default=False,
                        help="Flag T=upload training to the ftp server F=don't upload")
    parser.add_argument("-ct", "--checkpoint-timer", type=int, default=1800,
                        help="time threshhold to store the training in the dataset.(seconds)")
    parser.add_argument("-X", "--skip-training", type=str2bool, default=False,
                        help="Avoid training and only eval")

    return parser.parse_args()

# CONSTANST

args = process_command_line()
EPOCHS = args.epochs
MODEL_PATH = './{}-ds{}.pth'.format(args.net, args.dataset)
EPOCHS = args.epochs
BATCH = args.batch
DEEPVESSEL =False

if args.pre_transform:
    if args.dataset.startswith('G'):
        pre_transform = Crop(30,150,256,256)
    else:
        pre_transform = CropVessel12(30, 150, 256, 256)
else:
    pre_transform = None


if args.dataset == 'MNIST':
    dataset = MNIST(background=args.background)
elif args.dataset == 'GMNIST':
    dataset = GMNIST(background=args.background)
elif args.dataset == 'VESSEL12':
    dataset = VESSEL12(data_dir=args.vesseldir, pre_transform=pre_transform)
elif args.dataset == 'GVESSEL12':
    dataset = GVESSEL12(data_dir=args.vesseldir, pre_transform=pre_transform)
elif args.dataset == 'SVESSEL':
    dataset = SVESSEL(data_dir=args.svesseldir)
elif args.dataset == 'GSVESSEL':
    dataset = GSVESSEL(data_dir=args.svesseldir)

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
elif args.net == 'GFCND':
    model = GFCND()
elif args.net=='PointNet':
    model = PointNet()
elif args.net == 'UNet':
    model = UNet(n_channels=1, n_classes=1)
elif args.net == 'FCN':
    model = FCN(n_channels=1, n_classes=1)
elif args.net == 'DeepVessel':
    model = DeepVessel(dim=2, nchannels=1, nlabels=2)
    DEEPVESSEL =True
else:
    model = GFCNA()

if args.criterion == 'BCE':
    criterion = nn.BCELoss()
    sigmoid=False
elif args.criterion == 'BCElogistic':
    criterion = nn.BCEWithLogitsLoss()# criterion accepts logit. network produce logit
    sigmoid = True# evaluation flag to comput sigmoid because model output logit
elif args.criterion == 'DCS':
    criterion = DCS() # DCS assume network computes prob.
    sigmoid = False # not necesary to compute the signmout in the evaluation
elif args.criterion == 'DCSsigmoid':
    criterion = DCS(pre_sigmoid=True) # criterion accepts logit. network produce logit
    sigmoid = True # evaluation flag to comput sigmoid because model output logit
else:
    criterion = nn.BCELoss()
    sigmoid = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

if args.dataset[0] == 'G':
    trainer = Trainer(model=model,dataset=dataset, batch_size=BATCH,to_tensor=False, device=device, criterion=criterion)
    evaluator = Evaluator(dataset=dataset, batch_size=BATCH, to_tensor=False, device=device, sigmoid=sigmoid)
    trainer.load_model(model, MODEL_PATH)
elif args.net == 'DeepVessel':
    trainer = KTrainer(model=model, dataset=dataset, batch_size=BATCH)
    evaluator = KEvaluator(dataset)
    trainer.load_model(model,MODEL_PATH)
    model = trainer.model
else:
    trainer = Trainer(model=model, dataset=dataset, batch_size=BATCH, device=device, criterion=criterion)
    evaluator = Evaluator(dataset=dataset, batch_size=BATCH, device=device, sigmoid=sigmoid)
    trainer.load_model(model, MODEL_PATH)

model = model.to(device) if not DEEPVESSEL else model



def train(lr=0.001, progress_bar=False, fig_dir='./figs',prefix='NET'):
    loss_all, DCS, P, A, R, loss_epoch = trainer.load_checkpoint(prefix=prefix)
    timer = Timer(args.checkpoint_timer)
    for e in trainer.get_range(EPOCHS):
        model.train() if not DEEPVESSEL else None
        print('lesn loss all', len(loss_all), 'ken los all one elemtn', type(loss_all))

        loss = trainer.train_epoch(lr=lr, progress_bar=progress_bar)
        mean_loss = np.array(loss).mean()
        loss_epoch.append(mean_loss)
        print('EPOCH ', e, 'loss epoch', mean_loss)
        print('lesn loss all', len(loss_all), 'ken los all one elemtn', type(loss_all))
        print('len loss', len(loss), 'lesn loss one element', type(loss))
        new_loss = loss_all + loss
        loss_all = new_loss
        if DEEPVESSEL:
            print('Evaluation Epoch {}/{}...'.format(e, EPOCHS))
            DCS.append(evaluator.DCM(model, progress_bar=progress_bar))
            a, p, r = evaluator.bin_scores(model, progress_bar=progress_bar)
            P.append(p)
            A.append(a)
            R.append(r)
            print('DCS score:', DCS[-1], 'accuracy ', a, 'precision', p, 'recall', r)
        else:
            with torch.no_grad():
                print('Evaluation Epoch {}/{}...'.format(e,EPOCHS))
                model.eval()
                DCS.append(evaluator.DCM(model, progress_bar=progress_bar))
                a, p, r  = evaluator.bin_scores(model, progress_bar=progress_bar)
                P.append(p)
                A.append(a)
                R.append(r)
                print('DCS score:', DCS[-1], 'accuracy ', a, 'precision', p, 'recall', r )
        if timer.is_time():
            measurements = np.array([DCS, P, A, R, loss_epoch])
            trainer.save_model(MODEL_PATH)
            trainer.save_checkpoint(np.array(loss_all), measurements, prefix,  lr, args.dataset, e, EPOCHS, fig_dir, args.upload)
    loss_all = np.array(loss_all)
    measurements = np.array([DCS, P, A, R, loss_epoch])
    trainer.save_model(MODEL_PATH)
    trainer.save_checkpoint(loss_all, measurements, prefix,  lr, args.dataset, EPOCHS, EPOCHS, fig_dir, args.upload)


def eval(lr=0.001, progress_bar=False, fig_dir='./figs',prefix='NET'):
    model.eval() if not DEEPVESSEL else None
    # print('DCM factor: ' , evaluator.DCM(model, progress_bar=progress_bar))
    print('plotting one prediction')
    fig = evaluator.plot_prediction(model=model)
    savefigs(fig_name='{}_e{}_lr{}_ds{}_performance'.format(prefix,EPOCHS, lr, args.dataset),fig_dir=fig_dir, fig=fig)
    # plt.show()

if not args.skip_training:
    train(lr=args.lr, progress_bar=args.progressbar, fig_dir=args.figdir, prefix=args.net)
if DEEPVESSEL:
    model = trainer.model

eval(lr=args.lr, progress_bar=args.progressbar, fig_dir=args.figdir, prefix=args.net)