import torch
from matplotlib import pyplot as plt
from torch import nn

from lib.models import GFCN, GFCNA, GFCNB, GFCNC, GFCND, GFCNE, GFCNF, GFCNG, PointNet, UNet, FCN
from lib.process import Trainer, Evaluator, KTrainer, KEvaluator, DCS, TrainingDir
from lib.process.losses import estimatePositiveWeight, GeneralizedDiceLoss, FocalLoss, DiceLoss
from lib.utils import savefigs


from lib.datasets import GMNIST, GSVESSEL, GVESSEL12, GISLES2018
from lib.datasets.gisles2018 import isles2018_reshape as gisles2018_reshape
from lib.datasets.gisles2018 import get_modalities as gisles_get_modalities

from lib.datasets import MNIST, VESSEL12, SVESSEL, Crop, CropVessel12, ISLES2018
from lib.datasets.isles2018 import get_modalities as isles_get_modalities
from lib.datasets.isles2018 import isles2018_reshape
try:
    from dvn import FCN as DeepVessel
except Exception as e:
    print('Warning: No module dvn. Failed to import deep vessel models, Exception: ', str(e))

def get_pretransform(*args, **kwargs):
    pre_transform_name = kwargs['pre_transform']
    dataset_name = kwargs['dataset']
    if pre_transform_name:
        if dataset_name.startswith('G'):
            pre_transform = Crop(30, 150, 256, 256)
        else:
            pre_transform = CropVessel12(30, 150, 256, 256)
    else:
        pre_transform = None
    return pre_transform

def get_dataset(*args, **kwargs):
    dataset_name = kwargs['dataset'],
    pre_trasform_name = kwargs['pre_transform']
    pre_transform = get_pretransform(pre_trasform=pre_trasform_name)
    modalites = get_modalities(dataset_name=dataset_name, modalities=kwargs['modalities'])
    if dataset_name == 'MNIST':
        dataset = MNIST(background=kwargs['background'])
        reshape_transform = None
    elif dataset_name == 'GMNIST':
        dataset = GMNIST(background=kwargs['background'])
        reshape_transform = None
    elif dataset_name == 'VESSEL12':
        dataset = VESSEL12(data_dir=kwargs['vesseldir'], pre_transform=pre_transform)
        reshape_transform = None
    elif dataset_name == 'GVESSEL12':
        dataset = GVESSEL12(data_dir=kwargs['vesseldir'], pre_transform=pre_transform)
        reshape_transform = None
    elif dataset_name == 'SVESSEL':
        dataset = SVESSEL(data_dir=kwargs['svesseldir'])
        reshape_transform = None
    elif dataset_name == 'GSVESSEL':
        dataset = GSVESSEL(data_dir=kwargs['svesseldir'])
        reshape_transform = None
    elif dataset_name == 'GENDOSTROKE':
        dataset = GENDOSTROKE(data_dir=kwargs['endodir'])
        reshape_transform = None  # TODO: this is missing endostroke_reshape
    elif dataset_name == 'GISLES2018':
        dataset = GISLES2018(data_dir=kwargs['islesdir'], modalities=modalites, useful=kwargs['useful'], fold=kwargs['fold'])
        reshape_transform = gisles2018_reshape
    elif dataset_name == 'ISLES2018':
        dataset = ISLES2018(data_dir=kwargs['islesdir'], modalities=modalites, useful=kwargs['useful'], fold=kwargs['fold'])
        reshape_transform = isles2018_reshape
    else:
        dataset = MNIST()
        reshape_transform = None
    return dataset, reshape_transform

def get_modalities(dataset_name, modalities):
    if dataset_name == 'GISLES2018':
        return gisles_get_modalities(modalities)
    elif dataset_name == 'ISLES2018':
        return isles_get_modalities(modalities)
    else:
        return None


def get_model(net, postnorm, pweights, mod, dataset):
    _modalities = get_modalities(dataset_name=dataset, modalities=mod)
    NUM_INPUTS = 1 if _modalities is None else len(_modalities)
    if net == 'GFCN':
        model = GFCN(input_channels=NUM_INPUTS)
    elif net == 'GFCNA':
        model = GFCNA(input_channels=NUM_INPUTS, postnorm_activation=postnorm, weight_upool=pweights)
    elif net == 'GFCNB':
        model = GFCNB(input_channels=NUM_INPUTS, postnorm_activation=postnorm, weight_upool=pweights)
    elif net == 'GFCNC':
        model = GFCNC(input_channels=NUM_INPUTS, postnorm_activation=postnorm, weight_upool=pweights)
    elif net == 'GFCND':
        model = GFCND(input_channels=NUM_INPUTS)
    elif net == 'GFCNE':
        model = GFCNE(input_channels=NUM_INPUTS, postnorm_activation=postnorm)
    elif net == 'GFCNF':
        model = GFCNF(input_channels=NUM_INPUTS, postnorm_activation=postnorm)
    elif net == 'GFCNG':
        model = GFCNG(input_channels=NUM_INPUTS)
    elif net == 'PointNet':
        model = PointNet(input_channels=NUM_INPUTS)
    elif net == 'UNet':
        model = UNet(n_channels=NUM_INPUTS, n_classes=1)
    elif net == 'FCN':
        model = FCN(n_channels=NUM_INPUTS, n_classes=1)
    elif net == 'DeepVessel':
        model = DeepVessel(dim=2, nchannels=NUM_INPUTS, nlabels=2)
        DEEPVESSEL = True
    else:
        raise ValueError('Unknown network')
    return model

def get_criterion(criterion_name, progressbar, dataset=None, weight = None):
    assert dataset is not None and weight is None, 'Dataset must be specified when weight is not given'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if criterion_name == 'BCE':
        criterion = nn.BCELoss()  # criterion accepts probabilities, we assume that the network outputs prob
        sigmoid = False  # therefore, we don't calculate sigmoid during evaluation, we set eval flag to zero.
    elif criterion_name == 'BCElogistic':
        criterion = nn.BCEWithLogitsLoss()  # criterion accepts logit. network produce logit
        sigmoid = True  # evaluation flag to comput sigmoid because model output logit
    elif criterion_name == 'DCS':
        criterion = DCS()  # DCS assume network computes prob.
        sigmoid = False  # not necesary to compute the signmout in the evaluation
    elif criterion_name == 'DCSsigmoid':
        criterion = DCS(pre_sigmoid=True)  # criterion accepts logit. network produce logit
        sigmoid = True  # evaluation flag to comput sigmoid because model output logit
    elif criterion_name == 'BCEweightedlogistic':
        if weight is None:
            pos_weight = estimatePositiveWeight(dataset.train, progress_bar=progressbar)
        else:
            pos_weight = weight
        pos_weight = torch.tensor([pos_weight])
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight.to(device))  # criterion accepts logit. network produce logit
        sigmoid = True  # evaluation flag to comput sigmoid because model output logit
    elif criterion_name == 'GDL':
        criterion = GeneralizedDiceLoss()  # criterion accepts probability
        sigmoid = False  # not necesary to compute the sigmoid, because model output probability
    elif criterion_name == 'GDLsigmoid':
        criterion = GeneralizedDiceLoss(pre_sigmoid=True)  # criterion accepts probability
        sigmoid = True  # not necesary to compute the sigmoid, because model output probability
    elif criterion_name == 'FL':
        criterion = FocalLoss()  # criterion accepts probability
        sigmoid = False  # not necesary to compute the sigmoid, because model output probability
    elif criterion_name == 'FLsigmoid':
        criterion = FocalLoss(pre_sigmoid=True)  # criterion accepts probability
        sigmoid = True  # not necesary to compute the sigmoid, because model output probability
    elif criterion_name == 'DL':
        criterion = DiceLoss()  # criterion accepts probability
        sigmoid = False  # not necesary to compute the sigmoid, because model output probability
    elif criterion_name == 'DLsigmoid':
        criterion = DiceLoss(pre_sigmoid=True)  # criterion accepts probability
        sigmoid = True  # not necesary to compute the sigmoid, because model output probability
    else:
        criterion = nn.BCELoss()
        sigmoid = False
    return criterion, sigmoid


def get_evaluators(model, dataset, net, batch, **kwargs):
    criterion, sigmoid = get_criterion(criterion_name=kwargs['criterion'],
                                       progressbar=kwargs['progressbar'],
                                       dataset = dataset,
                                       weight=kwargs['weight'])
    dataset_name = kwargs['dataset']
    net_name = kwargs['net']
    BATCH = kwargs['batch']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TRAINING_DIR = TrainingDir(kwargs['training_dir'], kwargs['net'], kwargs['dataset'], kwargs['id'], kwargs['epochs'], kwargs['load_model'])
    if kwargs['dataset'][0] == 'G':
        trainer = Trainer(model=model, dataset=dataset, batch_size=BATCH, to_tensor=False, device=device,
                          criterion=criterion, sigmoid=sigmoid)
        evaluator_val = Evaluator(dataset=dataset, batch_size=BATCH, to_tensor=False, device=device, sigmoid=sigmoid,
                                  eval=True, criterion=criterion)
        evaluator_test = Evaluator(dataset=dataset, batch_size=BATCH, to_tensor=False, device=device, sigmoid=sigmoid,
                                   criterion=criterion)
        trainer.load_model(model, TRAINING_DIR.model_path)
    elif kwargs['net'] == 'DeepVessel':
        trainer = KTrainer(model=model, dataset=dataset, batch_size=BATCH)
        evaluator_val = KEvaluator(dataset, eval=True, criterion=criterion)
        evaluator_test = KEvaluator(dataset, criterion=criterion)
        trainer.load_model(model, TRAINING_DIR.model_path)
        model = trainer.model
    elif kwargs['dataset'] == 'ISLES2018':
        trainer = Trainer(model=model, dataset=dataset, batch_size=BATCH, to_tensor=False, device=device,
                          criterion=criterion,
                          sigmoid=sigmoid)
        evaluator_val = Evaluator(dataset=dataset, batch_size=BATCH, to_tensor=False, device=device, sigmoid=sigmoid,
                                  eval=True,
                                  criterion=criterion)
        evaluator_test = Evaluator(dataset=dataset, batch_size=BATCH, to_tensor=False, device=device, sigmoid=sigmoid,
                                   criterion=criterion)
        trainer.load_model(model, TRAINING_DIR.model_path)
    else:
        trainer = Trainer(model=model, dataset=dataset, batch_size=BATCH, device=device, criterion=criterion,
                          sigmoid=sigmoid)
        evaluator_val = Evaluator(dataset=dataset, batch_size=BATCH, device=device, sigmoid=sigmoid, eval=True,
                                  criterion=criterion)
        evaluator_test = Evaluator(dataset=dataset, batch_size=BATCH, device=device, sigmoid=sigmoid,
                                   criterion=criterion)
        trainer.load_model(model, TRAINING_DIR.model_path)
    return trainer, evaluator_val, evaluator_test, TRAINING_DIR


def plot_sample_figs(_sample_to_plot, _case_id=None, *args, **kwargs):
    net, postnorm, pweights, mod, dataset_name = kwargs['net'], kwargs['postnorm'], kwargs['pweights'], kwargs['mod'], kwargs['dataset']
    model = get_model(**kwargs)
    _, _, evaluator_test, TRAINING_DIR = get_evaluators(model=model, **kwargs)
    dataset, reshape_transform = get_dataset(dataset=dataset_name, **kwargs)
    modalities = get_modalities(dataset_name=dataset_name, modalities=mod)
    fig_activation, case_id, N = evaluator_test.plot_graph(model=model, N=_sample_to_plot,
                                                           reshape_transform=reshape_transform,
                                                           modalities=modalities,
                                                           case_id=_case_id)

    savefigs(fig_name='{}_{}_{}_activation'.format(TRAINING_DIR.prefix, case_id, N), fig_dir=TRAINING_DIR.fig_dir,
             fig=fig_activation)
    plt.close()

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