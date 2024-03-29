import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from skimage import color, measure
from skimage.exposure import exposure
from torch_geometric.data import Data, Batch
from hausdorff import hausdorff_distance

from .losses import DiceCoeff, calculate_optimal_threshold, calculate_auc, check_label_not_unique, calculate_cod, \
    calculate_hausdorff_distance
from .progress_bar import printProgressBar
from torch import sigmoid
import numpy as np

from ..datasets.transforms import reshape_square
from ..graph.batch import to_torch_batch
from ..utils.csv import dict_to_csv


class MetricsLogs(object):
    def __init__(self, measurements, monitor_metric="DCM"):
        assert isinstance(measurements, list), "measurements must be a list"
        # metrics calculations configuration
        if measurements is None:
            self._measurements = ["train_loss",
                                  "val_loss",
                                  "DCM",]
        else:
            # additional metrics are introduced in measurements list.
            self._measurements = measurements
            self._measurements = list(set(self._measurements+["train_loss", "val_loss", "DCM"]))
        # used to track metric of last evaluation.
        self.best_metric = None
        self.current_metric = None
        self.recent_update = False
        self.monitor_metric = monitor_metric
        # List will be fill with loss per batch, every epoch produces a list that will extended here (logged)
        self._loss_per_iter = []
        # initialization of measurements collections. Measurements updated externally. The default metrics are:
        # "train_loss": [],
        # "val_loss": [],
        # "DCS": []}
        # additional metrics are introduced in measurements list.
        self._metric_logs = {}
        for m in self._measurements:
            self._metric_logs[m] = []

    def reset_measurements(self, measurements: dict):
        # updates the list of measurements but resets the current metric logs
        # measurements is dictionary of lists of measurements
        assert isinstance(measurements, dict), "Measurements must be a dictionary of lists of measurements"
        default_metrics = ["train_loss", "val_loss", "DCM"]
        assert all([d in measurements.keys() for d in
                    default_metrics]), "Failed to reset metrics, Measurements must contain default metrics"
        if not all([d in self._measurements for d in measurements.keys()]) \
            or not all([d in measurements.keys() for d in self._measurements]):
            print("Warning: reset metrics will overwrite existing metrics list with different values. Before: \n "
                  "{} \n After: {}".format(self._measurements, measurements.keys()))
        self._measurements = list(measurements.keys())
        self.best_metric = None
        self.current_metric = None
        self.recent_update = False
        self._metric_logs = measurements

    def get_binary_metrics(self):
        # Gets only binary metrics. Metrics is a dictionary
        # PPV stands for Positive Predicted Value
        # COD stands for Coeficient of Determination
        # AUC stands for area under the curve
        supported_binary_metrics = ["AUC", "accuracy", "recall", "precision", "PPV"]
        binary_metrics = {m: self._metric_logs[m] for m in supported_binary_metrics
                          if m in self._metric_logs.keys()}
        return binary_metrics

    def get_non_binary_metrics(self):
        # get only the non binary metrics.
        # ASSD stands for Avg.Sym.Surf.Dist (ASSD)
        supported_non_binary_metrics = ["train_loss", "val_loss", "DCM", "HD", "ASSD", "COD"]
        non_binary_metrics = {m: self._metric_logs[m] for m in supported_non_binary_metrics
                              if m in self._metric_logs.keys()}
        return non_binary_metrics

    def update_loss_log(self, loss_values: list):
        # loss_values are the outputs from the training epoch. A list of loss values per batch in the epoch.
        assert isinstance(loss_values, list), "loss_values input must be a list or a tuple"
        # updates the list loss per iteration, equivalent to the loss per step
        self._loss_per_iter.extend(loss_values)

    def update_measurement(self, metric_values):
        # accepts a metric dictionary to update values.
        assert all([k in self._measurements for k in metric_values.keys()]), \
            "All measurements must be updated at  same time. Given: {}, Expected: {},".format(metric_values.keys(),
                                                                                              self._measurements.keys())
        for t, m in metric_values.items():
            g = self._metric_logs[t]
            g.append(m)
            self._metric_logs[t] = g
            if t == self.monitor_metric:
                self._update_best_metric(m)

    def _update_best_metric(self, metric):
        # Updates the best metric
        self.current_metric = metric
        # it will occur only the first time when not metric was ever updated
        # NOTE: this can also be used to change the monitor metrics ... only works for the prev metric
        if self.best_metric is None:
            self.best_metric = self.current_metric
            self.recent_update = True
            return
        # updates best metric if current_metric is better
        if self.current_metric > self.best_metric:
            self.best_metric = self.current_metric
            self.recent_update = True
        else:
            self.recent_update = False

    def is_best_metric(self):
        return self.recent_update

    def get_measurements(self):
        return self._metric_logs

    def get_loss_per_iter(self):
        return self._loss_per_iter
def plot_graph(g, image=None, ax=None, channel=0, mag=1.0, th=None, figsize=(70,70)):
    """
    Plots a graph in a matplotlib figure.
    """
    assert channel in range(g.num_node_features), "Channel must be in range [0, {}]".format(g.num_node_features)
    assert mag > 0, "Magnification must be greater than 0"
    assert figsize[0] > 0, "Figure width must be greater than 0"
    assert figsize[1] > 0, "Figure height must be greater than 0"
    # create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    # gather graph data
    pos_x, pos_y = np.zeros(g.num_nodes), np.zeros(g.num_nodes)
    for k, g_pos in enumerate(g.pos):
        pos_x[k], pos_y[k] = g_pos[0]*mag, g_pos[1]*mag
    values = np.zeros(g.num_nodes)
    for k, g_value in enumerate(g.x[:, channel]):
        values[k] = g_value
    coo_matrix = g.edge_index
    # define image limits
    if image is not None:
        xmin, xmax, ymin, ymax = image.shape[0]-0.5, -0.5, image.shape[1]-0.5, -0.5
    else:
        xmin, xmax, ymin, ymax = -0.5, max(pos_x)+0.5, -0.5, max(pos_y)+0.5 
    # plot image
    if image is not None:
        image = image.copy()/image.max()
        image = color.gray2rgb(image)
        ax.imshow(image[::-1,:], cmap='gray')
    # plot graph
    for i in range(g.num_edges):
        ii, jj = int(coo_matrix[0, i]), int(coo_matrix[1, i])
        # filter repeated edges
        if ii == jj or ii > jj:
            continue
        if th is not None and (values[ii]<th or values[jj]<th):
            continue
        ax.plot([pos_x[coo_matrix[0, i]], pos_x[coo_matrix[1, i]]],
                [pos_y[coo_matrix[0, i]], pos_y[coo_matrix[1, i]]],
                c='lightgray', alpha=0.3,
                linewidth=0.1*figsize[1])
    colors = [cm.bwr(color) for color in values]
    for xx, yy, cc, vv in zip(pos_x, pos_y, colors, values):
        ax.plot(xx, yy, 'o', c=cc, markersize=0.2*figsize[1])
    ax.axis('scaled')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    # return figure
    return fig


class Evaluator(object):
    def __init__(self, dataset, batch_size=64, to_tensor=True, device=None, sigmoid=False,  eval=False, criterion=None):
        if eval:
            self.dataset = dataset.val
        else:
            self.dataset = dataset.test
        self._batch_size = batch_size
        self.dataset.enforce_batch(self._batch_size)
        self.to_tensor = to_tensor
        self.device = device if device is not None else torch.device('cpu')
        self.sigmoid = sigmoid
        # used for getting the validation/test loss
        self.criterion = criterion
        self.opt_th = 0.5

    def __len__(self):
        return len(self.dataset.get_all_cases_id())

    def DCM(self, model, progress_bar=True):
        DCM_accum = []
        N = 0
        L = self.dataset.num_batches
        if progress_bar:
            printProgressBar(0, L, prefix='DCM:', suffix='Complete', length=50)
        i = 0
        self.dataset.enforce_batch(self._batch_size)

        for image, label in self.dataset.batches():
            features = torch.tensor(image).float() if self.to_tensor else image
            label = torch.tensor(label).float() if self.to_tensor else label
            features = features.to(self.device)
            label = label.to(self.device)
            prediction = model(features)
            if isinstance(prediction, Data):
                prediction = to_torch_batch(prediction)
            pred_mask = (sigmoid(prediction) > self.opt_th).float() if self.sigmoid else (prediction > self.opt_th).float()
            # reorganize prediction according to the batch.
            if not pred_mask.size(0) == label.size(0):
                b = label.size(0)
                pred_mask = pred_mask.view(b, -1)
            if not isinstance(image, Data):
                b = label.shape[0]
                label = label.view(b, -1)
                pred_mask = pred_mask.view(b, -1)
            DCM_accum.append(dice_coeff(pred_mask, label).item())
            N += label.numel()
            if progress_bar:
                printProgressBar(i, L, prefix='DCM:', suffix='Complete', length=50)
            else:
                print('DCM Epoch: in batch ', i+1, ' out of ', L, '(percentage {}%)'.format(100.0*(i+1)/L))
            i += 1

        # self.dataset.enforce_batch(1)
        DCM = np.array(DCM_accum).mean()
        # if self.monitor_metric == "DCM":
        #     self.update_metric(DCM)
        return DCM

    def calculate_metric(self, model, progress_bar=False, metrics=('val_loss'), reshape_transform=None):
        if 'val_loss' in metrics and self.criterion is None:
            raise ValueError('Criterion must be specified in the instance of the object Evaluator')
        metrics_values = {m:[] for m in metrics if m != "train_loss"}
        L = self.dataset.num_batches
        prefix = f"Calculating metrics {metrics}: "
        if progress_bar:
            printProgressBar(0, L, prefix=prefix, suffix='Complete', length=50)
        i = 0
        sample = self.dataset[0]
        is_graph_tensor = isinstance(sample, (Data, Batch))
        del sample
        self.dataset.enforce_batch(self._batch_size)
        for image, label in self.dataset.batches():
            features = torch.tensor(image).float() if self.to_tensor else image
            label = torch.tensor(label).float() if self.to_tensor else label
            features = features.to(self.device)
            label = label.to(self.device)
            prediction = model(features)
            if is_graph_tensor:
                prediction = to_torch_batch(prediction)
            pred_mask = (sigmoid(prediction) > self.opt_th).float() if self.sigmoid else (
                    prediction > self.opt_th).float()
            # now converts the predictino from logits to probabilities if necessary.
            pred_prob = sigmoid(prediction) if self.sigmoid else prediction.clone()
            # reorganize prediction according to the batch.
            if not pred_mask.size(0) == label.size(0):
                b = label.size(0)
                pred_mask = pred_mask.view(b, -1)
                pred_prob = pred_prob.view(b, -1)
            # compute square versions of pred and label as np arrays to compute the HD distance
            if is_graph_tensor:
                if reshape_transform is None:
                    label_square = reshape_square(label)
                    pred_mask_square = reshape_square(pred_mask)
                else:
                    label_square = reshape_transform(label)
                    pred_mask_square = reshape_transform(pred_mask)
            elif reshape_transform is not None:
                label_square = reshape_transform(label)
                pred_mask_square = reshape_transform(pred_mask)
            else:
                label_square = label
                pred_mask_square = pred_mask
            # after computing the square np.arrays we flat the predictions to compute the other metrics
            if not is_graph_tensor:
                b = label.shape[0]
                label = label.view(b, -1)
                prediction = prediction.view(b, -1)
                pred_mask = pred_mask.view(b, -1)
                pred_prob = pred_prob.view(b, -1)
            for m in metrics:
                if m == 'val_loss':
                    g = metrics_values['val_loss']
                    g.append(self.criterion(prediction, label).item())
                    metrics_values['val_loss'] = g
                elif m == "DCM":
                    dcm = dice_coeff(pred_prob, label).item()
                    g = metrics_values["DCM"]
                    g.append(dcm)
                    metrics_values["DCM"] = g
                elif m == "HD":
                    hd = calculate_hausdorff_distance(pred_mask_square, label_square)
                    metrics_values["HD"].append(hd)
                elif m == "AUC":
                    auc = calculate_auc(pred_prob, label)
                    metrics_values["AUC"].append(auc.mean().item())
                elif m == "COD":
                    COD = calculate_cod(pred_prob, label).mean().item()
                    metrics_values["COD"].append(COD)
            if progress_bar:
                printProgressBar(i, L, prefix=prefix, suffix='Complete', length=50)
            else:

                if i % int(L/10) == 0 or i == 0:
                    print('Eval-Loss: in batch ', i+1, ' out of ', L, '(percentage {}%)'.format(100.0*(i+1)/L))
            i += 1
        print('metrics[DCM == : \n ', metrics_values["DCM"])
        metrics_avgs = {m: np.array(g).mean() for m, g in metrics_values.items()}
        return metrics_avgs

    def bin_scores(self, model, progress_bar=False, metrics=("accuracy","recall","precision")):
        # correct = 0
        # TP = 0
        # FP = 0
        # FN = 0
        # N = 0
        metric_values = {m: [] for m in metrics}
        eps = 0.0001
        L = self.dataset.num_batches
        prefix_text = f"Calculating binary metrics {metrics}: "
        if progress_bar:
            printProgressBar(0, L, prefix=prefix_text, suffix='Complete', length=50)
        i = 0
        self.dataset.enforce_batch(self._batch_size)
        for image, label in self.dataset.batches():
            # feature and label conversion
            features = torch.tensor(image).float() if self.to_tensor else image
            label = torch.tensor(label).float() if self.to_tensor else label
            # to device
            features = features.to(self.device)
            label = label.long().to(self.device)
            prediction = model(features)
            if isinstance(prediction, Data):
                prediction = to_torch_batch(prediction)
            # calculate AUC and optimal threshold.
            # AUC, optimal_threshold = calculate_auc(prediction, label)
            pred_mask = (sigmoid(prediction) > self.opt_th).long() if self.sigmoid else (prediction > self.opt_th).long()
            if not pred_mask.size(0) == label.size(0):
                b = label.size(0)
                pred_mask = pred_mask.view(b, -1)
            if len(label.shape)>2:
                b = label.size(0)
                label = label.view(b, -1)
                pred_mask = pred_mask.view(b, -1)
            correct = pred_mask.eq(label).sum(axis=1)
            N = label.eq(label).sum(axis=1).float()
            pos = torch.nonzero(label.eq(1), as_tuple=True)
            neg = torch.nonzero(label.eq(0), as_tuple=True)
            pred_mask = pred_mask
            def collect_true_values(values, coords):
                vals = torch.zeros_like(label).bool()
                vals[coords] = values
                return vals.sum(dim=1)
            TP = collect_true_values(pred_mask[pos].eq(label[pos]), pos).float()
            FN = collect_true_values(pred_mask[pos].ne(label[pos]), pos).float()
            FP = collect_true_values(pred_mask[neg].ne(label[neg]), neg).float()

            if "accuracy" in metrics:
                metric_values["accuracy"].append(torch.mean(correct/N).item())
            if "recall" in metrics:
                metric_values["recall"].append(torch.mean((TP)/(TP+FN+eps)).item())
            if "precision" in metrics:
                metric_values["precision"].append(torch.mean((TP)/(TP+FP+eps)).item())
            if "PPV" in metrics:
                metric_values["PPV"].append(torch.mean((TP+FN)/(N+eps)).item())
            if progress_bar:
                printProgressBar(i, L, prefix=prefix_text, suffix='Complete', length=50)
            else:
                if i % int(L/10) == 0 or i == 0:
                    print('Bin Scores: in batch ', i+1, ' out of ', L, '(Completed {}%)'.format(100.0*(i+1)/L))
            i += 1
        metrics_avgs = {m: np.array(g).mean() for m, g in metric_values.items()}
        return metrics_avgs

    def scores_volume(self, model, progress_bar=False, metrics=None, reshape_transform=None, path_to_csv=None):
        if metrics is None:
            print("Nothing to do metrics is None")
            return
        metric_values = {m:[] for m in metrics if m not in ["val_loss", "train_loss"]}

        # todo: validation los must match with crterioin
        prefix = f"Calculating metrics in volume: "
        L = len(self.dataset.get_all_cases_id())
        if progress_bar:
            printProgressBar(0, L, prefix=prefix, suffix='', length=25)
        i = 0  # counter for the progress bar
        # check if it is is_graph_tensor:
        sample = self.dataset[0]
        is_graph_tensor = isinstance(sample, (Data, Batch))
        for case_id in self.dataset.get_all_cases_id():
            # collecting predictions for case_id
            preds = []
            preds_prob = []
            masks = []
            for sample in self.dataset.get_by_case_id(case_id, useful=False):
                if is_graph_tensor:
                    # this graph tensor
                    image = sample
                    mask = sample.y
                    image['batch'] = torch.zeros(sample.x.shape[0]).long()
                else:
                    image, mask = sample[0], sample[1]
                    if isinstance(image, torch.Tensor):
                        image = image.unsqueeze(0)
                    else:
                        image = np.expand_dims(image, axis=0)

                features = torch.tensor(image).float() if self.to_tensor else image.clone()
                mask = torch.tensor(mask).float() if self.to_tensor else mask

                features = features.to(self.device)
                mask = mask.to(self.device)
                # uses the input image to predict
                prediction = model(features)
                if isinstance(prediction, Data):
                    # prediction = to_torch_batch(prediction)
                    prediction = prediction.x
                # now converts the predictino from logits to probabilities if necessary.
                pred_prob = sigmoid(prediction) if self.sigmoid else prediction.clone()
                if is_graph_tensor:
                    if reshape_transform is None:
                        mask = reshape_square(mask)
                        prediction = reshape_square(prediction)
                        pred_prob = reshape_square(pred_prob)
                    else:
                        mask = reshape_transform(mask.cpu().detach().numpy())
                        prediction = reshape_transform(prediction)
                        pred_prob = reshape_transform(pred_prob)
                else:
                    if reshape_transform is not None:
                        mask = reshape_transform(mask)
                        prediction = reshape_transform(prediction)
                        pred_prob = reshape_transform(pred_prob)

                preds.append(prediction)
                preds_prob.append(pred_prob)
                masks.append(mask)
            pred = np.stack(preds).astype(np.float32)
            pred_prob = np.stack(preds_prob).astype(np.float32)
            mask = np.stack(masks).astype(np.float32)
            pred_mask = (pred_prob > self.opt_th).astype(np.float32)
            TP = (pred_mask * mask).sum()
            FP = (1 * ((pred_mask - mask) > 0)).sum()
            FN = (1 * ((mask - pred_mask) > 0)).sum()
            N = mask.size
            TN = N - TP - FP -FN
            for m in metrics:
                if m == "DCM":
                    dcm = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else np.array(0)
                    metric_values["DCM"].append(dcm.item())
                elif m == "HD":
                    hd = calculate_hausdorff_distance(pred_mask, mask)
                    hd = float(hd)
                    metric_values["HD"].append(hd)
                elif m == "AUC":
                    auc = calculate_auc(pred_prob.reshape(1, -1), mask.reshape(1. - 1))
                    metric_values["AUC"].append(auc.mean().item())
                elif m == "COD":
                    cod = calculate_cod(pred_prob.reshape(1,-1), mask.reshape(1, -1))
                    metric_values["COD"].append(cod.mean().item())
                elif m == "accuracy":
                    accuracy = (TP + TN) / N
                    metric_values["accuracy"].append(accuracy.item())
                elif "recall" == m:
                    recall = TP / (TP + FN) if (TP + FN) != 0 else np.array(0)
                    metric_values["recall"].append(recall.item())
                elif "precision" == m:
                    precision = TP / (TP + FP) if (TP + FP) != 0 else np.array(0)
                    metric_values["precision"].append(precision.item())
                elif "PPV" == m:
                    metric_values["PPV"].append(((TP+FN)/N).item())
            # printing progress bar
            if progress_bar:
                printProgressBar(i, L, prefix=prefix, suffix='Complete', length=50)
            else:
                if i % int(L/10) == 0 or i == 0:
                    print(prefix, ': in case ', i+1, ' out of ', L, '(percentage {}%)'.format(100.0*(i+1)/L))
            i += 1
        if path_to_csv is not None:
            dict_to_csv(path_to_csv, metric_values, index=self.dataset.get_all_cases_id())
        metric_avgs = {m: np.array(g).mean() for m, g in metric_values.items()}
        return metric_avgs

    def plot_graph(self, model, figsize=(70,70), N=190, reshape_transform=None,
                        modalities=None, case_id=None):
        if case_id is not None:
            print('Warning: case id is given, then \'N\'', N, ' is ignored.')
            indices_by_case_id = self.dataset.get_indices_by_case_id(case_id, useful=False, relative_dataset=True )
            # gets the central sample
            assert len(indices_by_case_id) > 0, 'Something went wrong, indices by case is empty'
            # NOTE: The actual error is that the case_id is not the testing/validation dataset.
            # NOTE: N is the index the dataset, which varies between folds, we get case id instead
            N = indices_by_case_id[(len(indices_by_case_id)-1) // 2]
            print('new N=', N)
        else:
            case_id = self.dataset.get_case_id(N)

        # loading the image: it can be a numpy.ndarray or a Data/Batch object
        # image, mask = self.dataset.next_batch(1, shuffle=False) # selects an aleatory value from the dataset
        sample = self.dataset[N]
        is_graph_tensor = isinstance(sample, (Data, Batch))
        if is_graph_tensor:
            # this graph tensor
            image = sample
            mask = sample.y
            image['batch'] = torch.zeros(sample.x.shape[0]).long()
        else:
            image, mask = sample[0], sample[1]
            if isinstance(image, torch.Tensor):
                image = image.unsqueeze(0)
            else:
                image = np.expand_dims(image, axis=0)

        model_input = torch.tensor(image).float() if self.to_tensor else image.clone()
        model_input = model_input.to(self.device)
        print('prediction to activations in latent space...')
        model.set_only_activation(True)
        activations = model(model_input)
        model.set_only_activation(False)
        # computes mean of activations and keeps dimensions
        if is_graph_tensor:
            activations_mean = activations.clone()
            activations_mean.x = torch.mean(activations.x, dim=1, keepdim=True)
            activations_mean.x = (activations_mean.x - torch.min(activations_mean.x))\
                                 / (torch.max(activations_mean.x) - torch.min(activations_mean.x))
        else:
            activations = activations.cpu().detach().numpy()
            activations_mean = np.mean(activations, axis=1, keepdims=True)
            activations_mean = (activations_mean - np.min(activations_mean))\
                                 / (np.max(activations_mean) - np.min(activations_mean))
        # transform the graph tensor or euclidean tensor to square images
        if is_graph_tensor:
            if reshape_transform is None:
                mask = reshape_square(mask)
                channels = None if modalities is None else len(modalities)
                images = reshape_square(image.x, channels=channels)

            else:
                mask = reshape_transform(mask)
                channels = None if modalities is None else len(modalities)
                # takes the first channel if there is more than one modality
                images = reshape_transform(image.x, channels=channels)
        else:
            if reshape_transform is not None:
                # TODO: this might fail! if you try to run euclideans :(
                images = reshape_transform(image)
                mask = reshape_transform(mask)

        # plotting the images next to the activations
        height, width, _ = images.shape
        fig_width = figsize[0]
        fig_height = fig_width * (5*height) / (2*width)
        fig, axs = plt.subplots(nrows=5, ncols=2, squeeze=True, gridspec_kw={'wspace': 0, 'hspace':0},
                                figsize=(fig_width, fig_height))
        for i in range(5):
            for j in range(2):
               axs[i, j].axis('off')
        # find countours of the mask
        contours = measure.find_contours(mask, 0.5)
        # plot imagenes next to the activations
        for i, modality in enumerate(modalities):
            if modality == 'CBV':
                image_cbv = exposure.adjust_gamma(images[:, :, 2], 0.5)
                axs[i, 0].imshow(image_cbv, cmap='viridis')
                plot_graph(activations_mean, image=image_cbv, ax=axs[i, 1], th=0.5)
                axs[i, 1].plot(contours[0][:, 1], contours[0][:, 0], 'y', linewidth=0.1*fig_width)
            elif modality == 'CBF':
                image_cbf = exposure.adjust_gamma(images[:, :, 3], 0.5)
                axs[i, 0].imshow(image_cbf, cmap='viridis')
                plot_graph(activations_mean, image=image_cbf, ax=axs[i, 1], th=0.5)
                axs[i, 1].plot(contours[0][:, 1], contours[0][:, 0], 'y', linewidth=0.1*fig_width)
            elif modality == 'CTN':
                axs[i, 0].imshow(images[:, :, i], cmap='gray')
                plot_graph(activations_mean, image=images[:, :, i], ax=axs[i, 1], th=0.5)
                axs[i, 1].plot(contours[0][:, 1], contours[0][:, 0], 'y', linewidth=0.1*fig_width)
            else:
                axs[i, 0].imshow(images[:, :, i], cmap='viridis')
                plot_graph(activations_mean, image=images[:, :, i], ax=axs[i, 1], th=0.5)
                axs[i, 1].plot(contours[0][:, 1], contours[0][:, 0], 'y', linewidth=0.1*fig_width)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        return fig, case_id, N

    def plot_prediction(self, model, fig=None, figsize=(10,10), N=190, overlap=True, reshape_transform=None,
                        modalities=None, get_case=False, case_id=None, activation_map=False):
        if case_id is not None:
            print('Warning: case id is given, then \'N\'', N, ' is ignored.')
            indices_by_case_id = self.dataset.get_indices_by_case_id(case_id, useful=False, relative_dataset=True )
            # gets the central sample
            assert len(indices_by_case_id) > 0, 'Something went wrong, indices by case is empty'
            # NOTE: The actual error is that the case_id is not the testing/validation dataset.
            # NOTE: N is the index the dataset, which varies between folds, we get case id instead
            N = indices_by_case_id[(len(indices_by_case_id)-1) // 2]
            print('new N=', N)
        else:
            case_id = self.dataset.get_case_id(N)

        # loading the image: it can be a numpy.ndarray or a Data/Batch object
        # image, mask = self.dataset.next_batch(1, shuffle=False) # selects an aleatory value from the dataset
        sample = self.dataset[N]
        # print('----- nasty saving object')
        # jtorch.save(sample, f'sample_case_id_{case_id}_N_{N}.pt')
        # // raise Exception('stop here!')
        is_graph_tensor = isinstance(sample, (Data, Batch))
        if is_graph_tensor:
            # this graph tensor
            image = sample
            mask = sample.y
            image['batch']=torch.zeros(sample.x.shape[0]).long()
        else:
            image, mask = sample[0], sample[1]
            if isinstance(image, torch.Tensor):
                image = image.unsqueeze(0)
            else:
                image = np.expand_dims(image, axis=0)

        input = torch.tensor(image).float() if self.to_tensor else image.clone()
        input = input.to(self.device)
        print(' prediction normal as prob...')
        prediction = model(input)
        if is_graph_tensor:
            prediction =  prediction.x
        pred_mask = (sigmoid(prediction) > self.opt_th).float() if self.sigmoid else (prediction > self.opt_th).float()
        # after using prediction for calculating the mask then the prediction is transformed to prob,
        prediction = sigmoid(prediction) if self.sigmoid else prediction
        
        if activation_map:
            print('prediction to activations in latent space...')
            model.set_only_activation()
            input = torch.tensor(image).float() if self.to_tensor else image.clone()
            activations = model(input)
            if is_graph_tensor:
                activations = activations.x
            model.set_only_activation()
            print('activations: ', activations.shape)
        # converts to an square if necessary
        if is_graph_tensor:
            if reshape_transform is None:
                mask = reshape_square(mask)
                channels = None if modalities is None else len(modalities)
                image = reshape_square(image.x, channels=channels)
                # takes the fistt channel if more than one modality
                image = image if channels is None else image[...,0]
                prediction = reshape_square(prediction)
                pred_mask = reshape_square(pred_mask)
            else:
                mask = reshape_transform(mask)
                channels = None if modalities is None else len(modalities)
                # takes the first channel if there is more than one modality
                image = reshape_transform(image.x, channels=channels, channel=0)
                prediction = reshape_transform(prediction)
                pred_mask = reshape_transform(pred_mask)
        else:
            if reshape_transform is not None:
                image = reshape_transform(image, channel=0)
                mask = reshape_transform(mask)
                prediction = reshape_transform(prediction)
                pred_mask = reshape_transform(pred_mask)
        # plot input image
        if not fig:
            fig = plt.figure(figsize=figsize)

        if overlap:
            mask = mask.squeeze()
            pred_mask = pred_mask.squeeze()
            cmap_TP = ListedColormap([[73/255, 213/255, 125/255, 1]])
            cmap_FP = ListedColormap([[255/255, 101/255, 80/255, 1]])
            cmap_FN = ListedColormap([[15/255, 71/255, 196/255, 1]])
            TP = pred_mask*mask
            FP = 1*((pred_mask-mask) > 0)
            FN = 1*((mask-pred_mask) > 0)
            NN = prediction.size

            alpha = 0.5
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(image.copy().squeeze(), cmap='gray')
            masked = np.ma.masked_where(FP == 0, FP)
            ax.imshow(masked, cmap=cmap_FP, alpha=alpha)
            masked = np.ma.masked_where(FN == 0, FN)
            ax.imshow(masked, cmap=cmap_FN, alpha=alpha)
            masked = np.ma.masked_where(TP == 0, TP)
            ax.imshow(masked, cmap=cmap_TP, alpha=alpha)

            epsilon = 1e-10
            A = TP.sum()
            B = FP.sum()
            C = FN.sum()
            C = NN-A-B-C
            a = (A+C)/NN
            p = (A + epsilon)/(A+B+epsilon)
            r = (A+epsilon)/(A+C+epsilon)
            dcm = 2*(p*r+epsilon)/(p+r+epsilon)
            print('OVERLAY IMAGE STATS N=', N , 'case=', case_id, ' ==> Accuracy: ', a ,' Precision: ', p, ', Recall: ', r, 'Dice: ', dcm)
        else:
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3)
            ax4 = fig.add_subplot(2, 2, 4)
            ax1.imshow(image.copy().squeeze(),cmap='gray')
            ax1.set_title('original image')
            # plot p(y=1|X=x)
            ax2.imshow(prediction.squeeze(), cmap='gray')
            ax2.set_title('probability map')
            # plot mask
            ax3.imshow(mask.squeeze(),cmap='gray')
            ax3.set_title('ground truth mask')
            # plot prediction
            ax4.imshow(pred_mask.squeeze(),cmap='gray')
            ax4.set_title('predicted mask >0.5 prob')
        return fig, case_id, N

    def plot_volumen(self,model, index=0, overlap=True, reshape_transform=None, modalities=None):
        # index was the offset of the processed_files list to concatenate into a volume,
        # but we change it is behavior , so index points the case_id index (raw_files). For example, if dataset.test
        # has the cases 10,12,13 with corresponding processed_files = [20,21,21,23,45,46,100,101,102] then index=1 will
        # comput the volume for the processed files 45 and 46.
        images = []
        # FIXME: Euclidean datasets dont have this method implemented yet or other datasets.
        # extracts the case_id from the dataset that corresponds to the index given
        case_id = self.dataset.get_all_cases_id()[index]

        # check if it is is_graph_tensor:
        sample = self.dataset[0]
        is_graph_tensor = isinstance(sample, (Data, Batch))

        # computes predictions for all the volume
        for sample in self.dataset.get_by_case_id(case_id, useful=False):
            if is_graph_tensor:
                # this graph tensor
                image = sample
                mask = sample.y
                image['batch'] = torch.zeros(sample.x.shape[0]).long()
            else:
                image, mask = sample[0], sample[1]
                if isinstance(image, torch.Tensor):
                    image = image.unsqueeze(0)
                else:
                    image = np.expand_dims(image, axis=0)

            # makes a prediction for the image and generate the prediciont mask with boundary 0.5
            input = torch.tensor(image).float() if self.to_tensor else image.clone()
            input = input.to(self.device)
            prediction = model(input)
            # pred_mask = (sigmoid(prediction) > self.opt_th).float()
            if isinstance(prediction, (Data, Batch)):
                prediction = prediction.x
            pred_mask = (sigmoid(prediction) > self.opt_th).float() if self.sigmoid else (prediction > self.opt_th).float()
            prediction = sigmoid(prediction) if self.sigmoid else prediction

            # if overlap flag then creates a plot of three colors TP, FN and FP.
            if overlap:
                if is_graph_tensor:
                    if reshape_transform is None:
                        mask = reshape_square(mask)
                        pred_mask = reshape_square(pred_mask)
                    else:
                        mask = reshape_transform(mask.cpu().detach().numpy())
                        pred_mask = reshape_transform(pred_mask)
                else:
                    if reshape_transform is not None:
                        mask = reshape_transform(mask)
                        pred_mask = reshape_transform(pred_mask)
                TP = pred_mask*mask
                FP = 1*((pred_mask-mask) > 0)
                FN = 1*((mask-pred_mask) > 0)
                mix = TP+2*FP+3*FN
                images.append(mix.squeeze())
            else:
                # show predictions and input channels
                if is_graph_tensor:
                    if reshape_transform is None:
                        mask = reshape_square(mask)
                        channels = None if modalities is None else len(modalities)
                        image = reshape_square(image.x, channels=channels)
                        prediction = reshape_square(prediction)
                        pred_mask = reshape_square(pred_mask)
                    else:
                        mask = reshape_transform(mask)
                        channels = None if modalities is None else len(modalities)
                        # takes the first channel if there is more than one modality
                        image = reshape_transform(image.x, channels=channels)
                        prediction = reshape_transform(prediction)
                        pred_mask = reshape_transform(pred_mask)
                else:
                    # in the euclidean case we swap C,Y,X or C,H,W to Y,X,C or H,W,C
                    if reshape_transform is not None:
                        image = reshape_transform(image)
                        mask = reshape_transform(mask)
                        prediction = reshape_transform(prediction)
                        pred_mask = reshape_transform(pred_mask)
                # concatenate the modality channels and prediction results
                if len(image.shape) == 2:
                    image = np.stack([image, mask, prediction, pred_mask], axis=-1)
                else:
                    pred_results = np.stack([mask, prediction, pred_mask], axis=-1)
                    image = np.concatenate([image, pred_results], axis=-1)
                images.append(image)
        # the resulting image is Z,Y,X for overlap, and Z,Y,X,C for no overlap.
        result = np.stack(images).astype(np.float32)
        if overlap:
            # calculate binary stats for volume:
            num_tp = (result == 1).sum()
            num_fp = (result == 2).sum()
            num_fn = (result == 3).sum()
            num_tn = (result == 0).sum()
            dice = (2 * num_tp) / (2 * num_tp + num_fp + num_fn)
            precision = (num_tp) / (num_tp + num_fp)
            recall = (num_tp) / (num_tp + num_fn)
            accuracy = (num_tp + num_tn) / (num_tp + num_tn + num_fp + num_fn)
            print('OVERLAY VOLUME STATS (case = ',case_id,' )==> Accuracy: ',  accuracy,
                  ' Precision: ', precision, ', Recall: ', recall, 'Dice: ', dice)
        return result, case_id


class KEvaluator(Evaluator):
    def __init__(self, dataset, batch_size=64, to_tensor=True, device=None, sigmoid=False, eval=False, criterion=None):
        '''
        Keras addaptation evaluate model
        :param dataset:
        :param batch_size:
        :param to_tensor:
        :param device:
        :param sigmoid:
        '''
        super(KEvaluator, self).__init__(dataset, batch_size=batch_size, to_tensor=to_tensor, device=device, sigmoid=sigmoid, eval=eval, criterion=criterion)
    def bin_scores(self, model, progress_bar=False):
        correct = 0
        TP = 0
        FP = 0
        FN = 0
        N = 0
        eps = 0.0001
        L = self.dataset.num_batches

        if progress_bar:
            printProgressBar(0, L, prefix='Binary Scores:', suffix='Complete', length=50)
        i = 0
        for image, label in self.dataset.batches():
            # feature and label conversion
            # features = torch.tensor(image).float()
            label = torch.tensor(label).float()
            # to device
            # features = features.to(self.device)
            label = label.float().to(self.device)
            prediction = model.predict(x=image)
            pred = torch.tensor(prediction).to(self.device)
            pred = (pred[:,1,:,:] > 0.5).float()

            if not pred.size(0) == label.size(0):
                b = label.size(0)
                pred = pred.view(b, -1)
            if len(label.shape) > 2:
                b = label.size(0)
                label = label.view(b, -1)
                pred = pred.view(b, -1)
            correct += pred.eq(label).sum().item()
            N += label.numel()
            mask_pos = label.eq(1).squeeze().nonzero()
            mask_neg = label.eq(0).squeeze().nonzero()
            TP += pred[:, mask_pos].eq(label[:, mask_pos]).sum().item()
            FP += pred[:, mask_pos].ne(label[:, mask_pos]).sum().item()
            FN += pred[:, mask_neg].ne(label[:, mask_neg]).sum().item()

            if progress_bar:
                printProgressBar(i, L, prefix='Acc, Rec, Pre:', suffix='Complete', length=50)
            else:
                if (i+1)%10==0:
                    print('Bin Scores: in batch ', i+1, ' out of ', L, '(Completed {}%)'.format(100.0*(i+1)/L))
            i += 1
        return correct/N, TP/(TP+FP+eps), TP/(TP+FN+eps)

    def DCM(self, model, progress_bar=False):
        DCM_accum = []
        N = 0
        L = self.dataset.num_batches
        if progress_bar:
            printProgressBar(0, L, prefix='DCM:', suffix='Complete', length=50)
        i = 0
        for image, label in self.dataset.batches():
            # features = torch.tensor(image).float() if self.to_tensor else image
            label = torch.tensor(label).float()
            # features = features.to(self.device)
            label = label.to(self.device)
            prediction = model.predict(x=image)
            pred_mask = torch.tensor(prediction).to(self.device)
            pred_mask = (pred_mask[:,1,:,:] > super().opt_th).float()

            # reorganize prediction according to the batch.
            if not pred_mask.size(0) == label.size(0):
                b = label.size(0)
                pred_mask = pred_mask.view(b, -1)
            DCM_accum.append(dice_coeff(pred_mask, label).item())
            N += label.numel()
            if progress_bar:
                printProgressBar(i, L, prefix='DCM:', suffix='Complete', length=50)
            else:
                if (i+1)%10 == 0:
                    print('DCM Epoch: in batch ', i+1, ' out of ', L, '(percentage {}%)'.format(100.0*(i+1)/L))
            i += 1

        # self.dataset.enforce_batch(1)

        return np.array(DCM_accum).mean()

    def plot_prediction(self,model, index=0, fig=None, figsize=(10,10), N=190, overlap=False):
        if not fig:
            fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        # loading the image: it can be a numpy.ndarray or a Data/Batch object
        image, mask = self.dataset.next_batch(1, shuffle=True) # selects an aleatory value from the dataset

        # input = torch.tensor(image).float() if self.to_tensor else image.clone()
        # input = input.to(self.device)
        # prediction = model(input)
        prediction = model.predict(x=image)[:,1,:,:]
        # pred_mask = (sigmoid(prediction) > 0.5).float()
        pred_mask = (prediction > super().opt_th)

        # if not isinstance(image,np.ndarray):
        #     dimension = image.x.size(0)# it will assume a square image, though we need a transformer for that
        #     dimension = np.sqrt(dimension).astype(int)
        #     mask = mask.cpu().detach().numpy().reshape((dimension,dimension))
        #     image = image.x.cpu().detach().numpy().reshape((dimension, dimension))
        #     prediction = torch.sigmoid(prediction.reshape((dimension, dimension)))
        #     pred_mask = pred_mask.reshape((dimension,dimension))
        #

        # plot input image
        #TODO: image will change its shape I need a transformer class

        ax1.imshow(image.copy().squeeze(),cmap='gray')
        ax1.set_title('original image')
        # plot p(y=1|X=x)
        ax2.imshow(prediction.squeeze(), cmap='gray')
        ax2.set_title('probability map')
        # plot mask
        ax3.imshow(mask.squeeze(),cmap='gray')
        ax3.set_title('ground truth mask')
        # plot prediction
        ax4.imshow(pred_mask.squeeze(),cmap='gray')
        ax4.set_title('predicted mask >0.5 prob')
        plt.show()
        return fig


def dice_coeff(inputs, target):
    """Dice coeff for batches"""
    if inputs.is_cuda:
        s = torch.tensor(1).float().cuda().zero_()
    else:
        s = torch.tensor(1).float().zero_()

    for i, c in enumerate(zip(inputs, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s/(i+1)

