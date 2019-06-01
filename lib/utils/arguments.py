import argparse

from config import VESSEL_DIR, ENDOSTROKE_DIR, ISLES2018_DIR


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
    parser.add_argument('--actions', '-A', nargs='+', type=str, default=['plot_samples', 'plot_vols', 'eval_samples', 'eval_vols'],
                        help='actions to be performed. Options are plot_samples, plot_vols,  eval_samples, eval_vols')
    return parser.parse_args()
