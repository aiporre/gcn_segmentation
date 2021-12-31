import argparse
import os.path
from tifffile import tifffile

from lib.process.evaluation import MetricsLogs
from lib.process.losses import estimatePositiveWeight, GeneralizedDiceLoss, FocalLoss, DiceLoss

try:
    from lib.datasets.gendostroke import GENDOSTROKE, endostroke_reshape
except Exception as e:
    print('Warning: No module torch geometric. Failed to import GENDOSTROKE, Exception: ', str(e))

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
    from lib.datasets import GISLES2018
    from lib.datasets.gisles2018 import isles2018_reshape as gisles2018_reshape
    from lib.datasets.gisles2018 import get_modalities as gisles_get_modalities
except Exception as e:
    print('Warning: No module torch geometric. Failed to import GISLES2018, Exception: ', str(e))

try:
    from lib.models import GFCN, GFCNA, GFCNC, GFCNB, PointNet, GFCND, GFCNE, GFCNG, GFCNF
except Exception as e:
    print('Warning: No module torch geometric. Failed to import models, Exception: ', str(e))

try:
    from dvn import FCN as DeepVessel
except Exception as e:
    print('Warning: No module dvn. Failed to import deep vessel models, Exception: ', str(e))

from lib.models import UNet, FCN
from lib.datasets import MNIST, VESSEL12, SVESSEL, Crop, CropVessel12, ISLES2018
from lib.datasets.isles2018 import get_modalities as isles_get_modalities
from lib.datasets.isles2018 import isles2018_reshape

from lib.process import Trainer, Evaluator, DCS, KEvaluator, KTrainer, TrainingDir
import matplotlib.pyplot as plt
import torch
from torch import nn

from config import VESSEL_DIR, SVESSEL_DIR, ENDOSTROKE_DIR, ISLES2018_DIR
from lib.utils import savefigs, Timer
import numpy as np

try:
    from keras import backend as K
except:
    print('Cannot load keras. not installed, because it a legacy Vessel GFCN')


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
    parser.add_argument("-D", "--training-dir", type=str, default='./',
                        help="path to save models, checkpoints and figures")
    parser.add_argument("-ed", "--endodir", type=str, default=ENDOSTROKE_DIR,
                        help="endovascular dataset dir")
    parser.add_argument("-idir", "--islesdir", type=str, default=ISLES2018_DIR,
                        help="ISLES 2018  dataset dir")
    parser.add_argument("-b", "--batch", type=int, default=2,
                        help="batch size of trainer and evaluator")
    parser.add_argument("-s", "--dataset", type=str, default='MNIST',
                        help="dataset to be used. Options: (G)MNIST, (G)VESSEL12, (G)SVESSEL, GENDOSTROKE")
    parser.add_argument("--useful", type=str2bool, default=False,
                        help="useful flag True activates filter, and only useful samples are collected in all datasets"
                             ". If False all samples are collected. Default is False")
    parser.add_argument("-f", "--fold", type=int, default=1,
                        help="Fold number that use test=23/train*=71=>train=65/val=6. "
                             "Number between 1 and 4. Defaults 1")
    parser.add_argument("--id", type=str, default='XYZ',
                        help="id for the training name")
    parser.add_argument("-n", "--net", type=str, default='GFCN',
                        help="network to be used. ....")
    parser.add_argument("--postnorm", type=str2bool, default=True,
                        help="Only in the GFCNx. If False, batch normalization is applied before the activation. "
                             "If True, batch, normalization is calculated after activation. Defaults True")
    parser.add_argument("-W", "--pweights", type=str2bool, default=False,
                        help="Activate proportional unpooing")
    parser.add_argument("--load-model", type=str, default='best',
                        help="loading model mode. Options are best, and last")
    parser.add_argument("-p", "--pre-transform", type=str2bool, default=False,
                        help="use a pre-transfrom to the dataset")
    parser.add_argument("-z", "--background", type=str2bool, default=True,
                        help="use a background in the MNIST dataset.")
    parser.add_argument("-mm", "--monitor-metric", type=str, default='DCM',
                        help="Monitor metric for saving models ")
    parser.add_argument("-c", "--criterion", type=str, default='BCE',
                        help="criterion: BCE or DCS or BCElogistic or DCSsigmoid or wBCElogistic or FL or FLsigmoid or "
                             "DL or DLsigmoid or GDL or GDLsigmoid")
    parser.add_argument("-w", "--weight", type=float, default=None,
                        help="Positive weight value for unbalanced datasets. If not given then it is estimated.")
    parser.add_argument("-u", "--upload", type=str2bool, default=False,
                        help="Flag T=upload training to the ftp server F=don't upload")
    parser.add_argument("-ct", "--checkpoint-timer", type=int, default=1800,
                        help="time threshhold to store the training in the dataset.(seconds)")
    parser.add_argument("-X", "--skip-training", type=str2bool, default=False,
                        help="Avoid training and only eval")
    parser.add_argument("-N", "--sample-to-plot", type=int, default=190,
                        help="sample to plot from the dataset")
    parser.add_argument("--mod", nargs="+", type=str, default=["CTN", "TMAX", "CBF", "CBV", "MTT"],
                        help=" Modalities for the ISLES2018 dataset. Defaults to [\"CTN\", \"TMAX\", \"CBF\", \"CBV\", \"MTT\"]")
    return parser.parse_args()


# CONSTANST

args = process_command_line()
print('=====================')
print('ARGUMENTS: ')
print(args)
print('=====================')
EPOCHS = args.epochs
TRAINING_DIR = TrainingDir(args.training_dir, args.net, args.dataset, args.id, EPOCHS, args.load_model)
TRAINING_DIR.makedirs()
print('Training Directory configuration is: ', str(TRAINING_DIR))
EPOCHS = args.epochs
BATCH = args.batch
DEEPVESSEL = False
MEASUREMENTS = ["train_loss", "val_loss", "DCM", 'accuracy', 'precision', 'recall', "HD", "COD", "PPV"]
if args.dataset == "GISLES2018":
    MODALITIES = gisles_get_modalities(args.mod)
elif args.dataset == 'ISLES2018':
    MODALITIES = isles_get_modalities(args.mod)
else:
    MODALITIES = None
NUM_INPUTS = 1 if MODALITIES is None else len(MODALITIES)

if args.pre_transform:
    if args.dataset.startswith('G'):
        pre_transform = Crop(30, 150, 256, 256)
    else:
        pre_transform = CropVessel12(30, 150, 256, 256)
else:
    pre_transform = None

if args.dataset == 'MNIST':
    dataset = MNIST(background=args.background)
    reshape_transform = None
elif args.dataset == 'GMNIST':
    dataset = GMNIST(background=args.background)
    reshape_transform = None
elif args.dataset == 'VESSEL12':
    dataset = VESSEL12(data_dir=args.vesseldir, pre_transform=pre_transform)
    reshape_transform = None
elif args.dataset == 'GVESSEL12':
    dataset = GVESSEL12(data_dir=args.vesseldir, pre_transform=pre_transform)
    reshape_transform = None
elif args.dataset == 'SVESSEL':
    dataset = SVESSEL(data_dir=args.svesseldir)
    reshape_transform = None
elif args.dataset == 'GSVESSEL':
    dataset = GSVESSEL(data_dir=args.svesseldir)
    reshape_transform = None
elif args.dataset == 'GENDOSTROKE':
    dataset = GENDOSTROKE(data_dir=args.endodir)
    reshape_transform = endostroke_reshape
elif args.dataset == 'GISLES2018':
    dataset = GISLES2018(data_dir=args.islesdir, modalities=MODALITIES, useful=args.useful, fold=args.fold)
    reshape_transform = gisles2018_reshape
elif args.dataset == 'ISLES2018':
    dataset = ISLES2018(data_dir=args.islesdir, modalities=MODALITIES, useful=args.useful, fold=args.fold)
    reshape_transform = isles2018_reshape
else:
    dataset = MNIST()
    reshape_transform = None

if args.net == 'GFCN':
    model = GFCN(input_channels=NUM_INPUTS)
elif args.net == 'GFCNA':
    model = GFCNA(input_channels=NUM_INPUTS, postnorm_activation=args.postnorm, weight_upool=args.pweights)
elif args.net == 'GFCNB':
    model = GFCNB(input_channels=NUM_INPUTS, postnorm_activation=args.postnorm, weight_upool=args.pweights)
elif args.net == 'GFCNC':
    model = GFCNC(input_channels=NUM_INPUTS, postnorm_activation=args.postnorm, weight_upool=args.pweights)
elif args.net == 'GFCND':
    model = GFCND(input_channels=NUM_INPUTS)
elif args.net == 'GFCNE':
    model = GFCNE(input_channels=NUM_INPUTS, postnorm_activation=args.postnorm)
elif args.net == 'GFCNF':
    model = GFCNF(input_channels=NUM_INPUTS, postnorm_activation=args.postnorm)
elif args.net == 'GFCNG':
    model = GFCNG(input_channels=NUM_INPUTS)
elif args.net == 'PointNet':
    model = PointNet()
elif args.net == 'UNet':
    model = UNet(n_channels=NUM_INPUTS, n_classes=1)
elif args.net == 'FCN':
    model = FCN(n_channels=NUM_INPUTS, n_classes=1)
elif args.net == 'DeepVessel':
    model = DeepVessel(dim=2, nchannels=NUM_INPUTS, nlabels=2)
    DEEPVESSEL = True
else:
    model = GFCNA()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if args.criterion == 'BCE':
    criterion = nn.BCELoss()  # criterion accepts probabilities, we assume that the network outputs prob
    sigmoid = False  # therefore, we don't calculate sigmoid during evaluation, we set eval flag to zero.
elif args.criterion == 'BCElogistic':
    criterion = nn.BCEWithLogitsLoss()  # criterion accepts logit. network produce logit
    sigmoid = True  # evaluation flag to comput sigmoid because model output logit
elif args.criterion == 'DCS':
    criterion = DCS()  # DCS assume network computes prob.
    sigmoid = False  # not necesary to compute the signmout in the evaluation
elif args.criterion == 'DCSsigmoid':
    criterion = DCS(pre_sigmoid=True)  # criterion accepts logit. network produce logit
    sigmoid = True  # evaluation flag to comput sigmoid because model output logit
elif args.criterion == 'BCEweightedlogistic':
    if args.weight is None:
        pos_weight = estimatePositiveWeight(dataset.train, progress_bar=args.progressbar)
    else:
        pos_weight = args.weight
    pos_weight = torch.tensor([pos_weight])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))  # criterion accepts logit. network produce logit
    sigmoid = True  # evaluation flag to comput sigmoid because model output logit
elif args.criterion == 'GDL':
    criterion = GeneralizedDiceLoss()  # criterion accepts probability
    sigmoid = False  # not necesary to compute the sigmoid, because model output probability
elif args.criterion == 'GDLsigmoid':
    criterion = GeneralizedDiceLoss(pre_sigmoid=True)  # criterion accepts probability
    sigmoid = True  # not necesary to compute the sigmoid, because model output probability
elif args.criterion == 'FL':
    criterion = FocalLoss()  # criterion accepts probability
    sigmoid = False  # not necesary to compute the sigmoid, because model output probability
elif args.criterion == 'FLsigmoid':
    criterion = FocalLoss(pre_sigmoid=True)  # criterion accepts probability
    sigmoid = True  # not necesary to compute the sigmoid, because model output probability
elif args.criterion == 'DL':
    criterion = DiceLoss()  # criterion accepts probability
    sigmoid = False  # not necesary to compute the sigmoid, because model output probability
elif args.criterion == 'DLsigmoid':
    criterion = DiceLoss(pre_sigmoid=True)  # criterion accepts probability
    sigmoid = True  # not necesary to compute the sigmoid, because model output probability
else:
    criterion = nn.BCELoss()
    sigmoid = False

model = model.to(device) if not DEEPVESSEL else model
if args.dataset[0] == 'G':
    trainer = Trainer(model=model, dataset=dataset, batch_size=BATCH, to_tensor=False, device=device,
                      criterion=criterion, sigmoid=sigmoid)
    evaluator_val = Evaluator(dataset=dataset, batch_size=BATCH, to_tensor=False, device=device, sigmoid=sigmoid,
                              eval=True, criterion=criterion)
    evaluator_test = Evaluator(dataset=dataset, batch_size=BATCH, to_tensor=False, device=device, sigmoid=sigmoid,
                               criterion=criterion)
    trainer.load_model(model, TRAINING_DIR.model_path)
elif args.net == 'DeepVessel':
    trainer = KTrainer(model=model, dataset=dataset, batch_size=BATCH)
    evaluator_val = KEvaluator(dataset, eval=True, criterion=criterion)
    evaluator_test = KEvaluator(dataset, criterion=criterion)
    trainer.load_model(model, TRAINING_DIR.model_path)
    model = trainer.model
elif args.dataset == 'ISLES2018':
    trainer = Trainer(model=model, dataset=dataset, batch_size=BATCH, to_tensor=False, device=device, criterion=criterion,
                      sigmoid=sigmoid)
    evaluator_val = Evaluator(dataset=dataset, batch_size=BATCH, to_tensor=False, device=device, sigmoid=sigmoid, eval=True,
                              criterion=criterion)
    evaluator_test = Evaluator(dataset=dataset, batch_size=BATCH, to_tensor=False, device=device, sigmoid=sigmoid, criterion=criterion)
    trainer.load_model(model, TRAINING_DIR.model_path)
else:
    trainer = Trainer(model=model, dataset=dataset, batch_size=BATCH, device=device, criterion=criterion,
                      sigmoid=sigmoid)
    evaluator_val = Evaluator(dataset=dataset, batch_size=BATCH, device=device, sigmoid=sigmoid, eval=True,
                              criterion=criterion)
    evaluator_test = Evaluator(dataset=dataset, batch_size=BATCH, device=device, sigmoid=sigmoid, criterion=criterion)
    trainer.load_model(model, TRAINING_DIR.model_path)


def train(lr=0.001, progress_bar=False):
    global model
    prefix_checkpoint = TRAINING_DIR.prefix
    eval_metric_logging = MetricsLogs(MEASUREMENTS, monitor_metric=args.monitor_metric)
    trainer.load_checkpoint(root=TRAINING_DIR.root, prefix=prefix_checkpoint, eval_logging=eval_metric_logging)
    timer = Timer(args.checkpoint_timer)
    for e in trainer.get_range(EPOCHS):
        trainer.model.train() if not DEEPVESSEL else None
        loss = trainer.train_epoch(lr=lr, progress_bar=progress_bar)
        mean_loss = np.array(loss).mean()
        eval_metric_logging.update_loss_log(loss)
        print('EPOCH ', e, 'loss epoch', mean_loss)
        model = trainer.model
        if DEEPVESSEL:
            print('Evaluation Epoch {}/{}...'.format(e, EPOCHS))
            # DCS.append(evaluator_val.DCM(model, progress_bar=progress_bar))
            dcs = evaluator_val.DCM(model, progress_bar=progress_bar)
            a, p, r = evaluator_val.bin_scores(model, progress_bar=progress_bar)
            val_loss = evaluator_val.calculate_metric(model, progress_bar=progress_bar,
                                                      reshape_transform=reshape_transform)
            print('DCS score:', dcs, 'accuracy ', a, 'precision ', p, 'recall ', r, 'val_loss ', val_loss)
        else:
            with torch.no_grad():
                print('Evaluation Epoch {}/{}...'.format(e, EPOCHS))
                model.eval()
                if e % int(EPOCHS / 10) == 0 or e == 0:
                    evaluator_val.opt_th = trainer.update_optimal_threshold(progress_bar=progress_bar)
                # DCS.append(evaluator_val.DCM(model, progress_bar=progress_bar))
                # DCM = evaluator_val.DCM(model, progress_bar=progress_bar)
                # include all the metrics calcuated using True Positive, False Negative and so on..
                binary_metrics_names = tuple(eval_metric_logging.get_binary_metrics().keys())
                binary_metrics = evaluator_val.bin_scores(model, progress_bar=progress_bar,
                                                          metrics=binary_metrics_names)
                # include all non binary metrics. for example val_loss,
                non_binary_metrics_names = tuple(eval_metric_logging.get_non_binary_metrics().keys())
                non_binary_metrics = evaluator_val.calculate_metric(model, progress_bar=progress_bar,
                                                                    metrics=non_binary_metrics_names,
                                                                    reshape_transform=reshape_transform)
                metrics = dict(non_binary_metrics, **binary_metrics)
                # metrics["DCM"]=DCM
                metrics["train_loss"] = mean_loss
                eval_metric_str = ""
                for m_name, m_value in metrics.items():
                    eval_metric_str += f" {m_name}={m_value} "
                print("Evaluation Metrics: ", eval_metric_str)
        # update metrics and loss logs in the trainer
        model = trainer.model
        eval_metric_logging.update_measurement(metrics)
        if eval_metric_logging.is_best_metric():
            print(
                'Saving new model: {} > {}'.format(eval_metric_logging.best_metric, eval_metric_logging.current_metric))
            trainer.save_model(TRAINING_DIR.model_path_best)
        if timer.is_time():
            trainer.save_checkpoint(TRAINING_DIR, lr, e, EPOCHS, eval_metric_logging, args.upload)
            trainer.save_model(TRAINING_DIR.model_path_last)

    # loss_all = np.array(loss_all)
    trainer.save_model(TRAINING_DIR.model_path)
    trainer.save_checkpoint(TRAINING_DIR, lr, e, EPOCHS, eval_metric_logging, args.upload)


def eval(progress_bar=False, modalities=None):
    model.eval() if not DEEPVESSEL else None
    eval_metric_logging = MetricsLogs(MEASUREMENTS, monitor_metric=args.monitor_metric)
    trainer.load_checkpoint(root=TRAINING_DIR.root, prefix=TRAINING_DIR.prefix, eval_logging=eval_metric_logging)
    evaluator_test.opt_th = 0.5  # :trainer.update_optimal_threshold(progress_bar=progress_bar)
    print('plotting one prediction')

    # Making the case if args.overlay_plot == True
    def plot_sample_vols(_sample_to_plot):
        # Ploting over lay volume
        overlay_vol, case_id = evaluator_test.plot_volumen(model=model, index=_sample_to_plot, overlap=True,
                                                           reshape_transform=reshape_transform, modalities=modalities)
        z, y, x = overlay_vol.shape[0], overlay_vol.shape[1], overlay_vol.shape[2]
        overlay_vol.tofile(os.path.join(TRAINING_DIR.fig_dir,
                                        '{}_vol_{}_{}x{}x{}.raw'.format(TRAINING_DIR.prefix, case_id, x, y, z)))
        # Plotting multi-channel volume.
        # Making the case with overlay_plot = False
        multichannel_vol, case_id = evaluator_test.plot_volumen(model=model, index=_sample_to_plot, overlap=False,
                                                                reshape_transform=reshape_transform,
                                                                modalities=modalities)
        z, y, x, c = multichannel_vol.shape[0], multichannel_vol.shape[1], multichannel_vol.shape[2], \
                     multichannel_vol.shape[3]
        multichannel_vol = multichannel_vol.transpose(0, 3, 1, 2)
        tiff_filename = '{}_vol_{}_{}x{}x{}x{}.tiff'.format(TRAINING_DIR.prefix, case_id, x, y, z, c)
        tifffile.imwrite(os.path.join(TRAINING_DIR.fig_dir, tiff_filename),
                         multichannel_vol, imagej=True, metadata={'axes': 'ZCYX'})

    def plot_sample_figs(_sample_to_plot, _case_id=None):
        fig_overlay_image, case_id, N = evaluator_test.plot_prediction(model=model, N=_sample_to_plot, overlap=True,
                                                                       reshape_transform=reshape_transform,
                                                                       modalities=modalities, get_case=True,
                                                                       case_id=_case_id)
        savefigs(fig_name='{}_{}_{}_overlap'.format(TRAINING_DIR.prefix, case_id, N), fig_dir=TRAINING_DIR.fig_dir,
                 fig=fig_overlay_image)
        plt.close()
        fig_four_plots, case_id, N = evaluator_test.plot_prediction(model=model, N=_sample_to_plot, overlap=False,
                                                                    reshape_transform=reshape_transform,
                                                                    modalities=modalities, get_case=True,
                                                                    case_id=_case_id)
        savefigs(fig_name='{}_{}_{}_performance'.format(TRAINING_DIR.prefix, case_id, N), fig_dir=TRAINING_DIR.fig_dir,
                 fig=fig_four_plots)
        plt.close()

    if args.sample_to_plot > 0:
        case_id_num = args.sample_to_plot
        plot_sample_figs(None, _case_id=evaluator_test.dataset.get_all_cases_id()[case_id_num])
        plot_sample_vols(args.sample_to_plot)
    else:
        total_test_samples = len(evaluator_test.dataset)
        print('plotting 2D all testing samples (', total_test_samples, ') ...')
        for n in range(total_test_samples):
            plot_sample_figs(n)
        print('plotting 3D all testing samples (', total_test_samples, ') ...')
        total_vol_samples = len(evaluator_test.dataset.get_all_cases_id())
        for c in range(total_vol_samples):
            plot_sample_vols(c)

        print('Done plotting')
    # plt.show()
    print('calculating stats...')
    metric_logs = MetricsLogs(MEASUREMENTS, monitor_metric=args.monitor_metric)
    binary_metrics_names = tuple(metric_logs.get_binary_metrics().keys())
    binary_metrics = evaluator_test.bin_scores(model, progress_bar=progress_bar,
                                               metrics=binary_metrics_names)
    # include all non binary metrics. for example val_loss,
    non_binary_metrics_names = list(metric_logs.get_non_binary_metrics().keys())
    non_binary_metrics_names.pop(non_binary_metrics_names.index("train_loss"))
    non_binary_metrics = evaluator_test.calculate_metric(model, progress_bar=progress_bar,
                                                         metrics=non_binary_metrics_names,
                                                         reshape_transform=reshape_transform)
    metrics = dict(non_binary_metrics, **binary_metrics)
    print('Calculated metrics testing set: \n', ''.join([f"{m} = {v}, " for m, v in metrics.items()]))

    metrics_vol = evaluator_test.scores_volume(model, progress_bar=progress_bar, metrics=MEASUREMENTS,
                                               reshape_transform=reshape_transform,
                                               path_to_csv=TRAINING_DIR.metrics_csv_path)

    print('Calculated metrics testing set per case: \n', ''.join([f"{m} = {v}, " for m, v in metrics_vol.items()]))


if not args.skip_training:
    train(lr=args.lr, progress_bar=args.progressbar)
if DEEPVESSEL:
    model = trainer.model
with torch.no_grad():
    eval(progress_bar=args.progressbar, modalities=MODALITIES)
