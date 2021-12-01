import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch_geometric.data import Data, Batch
from hausdorff import hausdorff_distance

from .losses import DiceCoeff, calculate_optimal_threshold, calculate_auc
from .progress_bar import printProgressBar
from torch import sigmoid
import numpy as np

from ..datasets.transforms import reshape_square
from ..graph.batch import to_torch_batch

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

    def update_optimal_threshold(self, model, progress_bar=True):
        opt_ths = []
        L = self.dataset.num_batches
        progress_bar_prefix = "Estimating optimal threshold"
        if progress_bar:
            printProgressBar(0, L, prefix=progress_bar_prefix, suffix='Complete', length=25)
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
            opt_ths.extend(calculate_optimal_threshold(prediction, label))
            if progress_bar:
                printProgressBar(i, L, prefix=progress_bar_prefix, suffix='Complete', length=50)
            else:
                if i % int(L / 10) == 0 or i == 0:
                    print(f'{progress_bar_prefix}: in batch ', i+1, ' out of ', L, '(percentage {}%)'.format(100.0*(i+1)/L))
            i += 1
        self.opt_th = np.array(opt_ths, dtype=np.float).mean().item()

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

    def calculate_metric(self, model, progress_bar=False, metrics=('val_loss')):
        if 'val_loss' in metrics and self.criterion is None:
            raise ValueError('Criterion must be specified in the instance of the object Evaluator')
        metrics_values = {m:[] for m in metrics if m != "train_loss"}
        L = self.dataset.num_batches
        prefix = f"Calculating metrics {metrics}: "
        if progress_bar:
            printProgressBar(0, L, prefix=prefix, suffix='Complete', length=50)
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
            for m in metrics:
                if m == 'val_loss':
                    g = metrics_values['val_loss']
                    g.append(self.criterion(prediction, label).item())
                    metrics_values['val_loss'] = g
                pred_mask = (sigmoid(prediction) > self.opt_th).float() if self.sigmoid else (prediction > self.opt_th).float()
                # now converts the predictino from logits to probabilities if necessary.
                prediction = sigmoid(prediction) if self.sigmoid else prediction

                # reorganize prediction according to the batch.
                if not pred_mask.size(0) == label.size(0):
                    b = label.size(0)
                    pred_mask = pred_mask.view(b, -1)
                if m == "DCM":
                    dcm = dice_coeff(pred_mask, label).item()
                    g = metrics_values["DCM"]
                    g.append(dcm)
                    metrics_values["DCM"] = g
                if m == "HD":
                    #hd = hausdorff_distance(prediction.detach().cpu().numpy().reshape(-1,1),
                    #                        label.detach().cpu().numpy().reshape(-1,1))
                    hd = 1 #float(hd)
                    g = metrics_values["HD"]
                    g.append(hd)
                    metrics_values["HD"] = g
                if m == "AUC":
                    auc = calculate_auc(prediction, label)
                    metrics_values["AUC"].append(auc.mean().item())
                if m == "COD":
                    eps = 1E-10
                    SS_res = torch.nn.functional.mse_loss(prediction, label)
                    pred_mean = pred_mask.mean(axis=0)
                    SS_var = torch.mean((label-pred_mean)**2)
                    COD = (1 - (SS_res+eps)/(SS_var+eps)).item()
                    metrics_values["COD"].append(COD)
            if progress_bar:
                printProgressBar(i, L, prefix=prefix, suffix='Complete', length=50)
            else:

                if i % int(L/10) == 0 or i == 0:
                    print('Eval-Loss: in batch ', i+1, ' out of ', L, '(percentage {}%)'.format(100.0*(i+1)/L))
            i += 1
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
            N = label.eq(label).sum(axis=1)
            pos = torch.nonzero(label.eq(1).squeeze(), as_tuple=True)
            neg = torch.nonzero(label.eq(0).squeeze(), as_tuple=True)
            def collect_true_values(values, coords):
                vals = torch.zeros_like(label).bool()
                vals[coords] = values
                return vals.sum(dim=1)
            TP = collect_true_values(pred_mask[pos].eq(label[pos]), pos)
            FN = collect_true_values(pred_mask[pos].ne(label[pos]), pos)
            FP = collect_true_values(pred_mask[neg].ne(label[neg]), neg)

            if "accuracy" in metrics:
                metric_values["accuracy"].append(torch.mean(correct/N+eps).item())
            if "recall" in metrics:
                metric_values["recall"].append(torch.mean((TP)/(TP+FN+eps)).item())
            if "precision" in metrics:
                metric_values["precision"].append(torch.mean((TP)/(TP+FP+eps)).item())
            if "PPV" in metrics:
                metric_values["PPV"].append(torch.mean((TP)/(N+eps)).item())
            if progress_bar:
                printProgressBar(i, L, prefix=prefix_text, suffix='Complete', length=50)
            else:
                if i % int(L/10) == 0 or i == 0:
                    print('Bin Scores: in batch ', i+1, ' out of ', L, '(Completed {}%)'.format(100.0*(i+1)/L))
            i += 1
        # metric_values = {}
        # if "accuracy" in metrics:
        #     metric_values["accuracy"] = correct/N
        # if "recall" in metrics:
        #     metric_values["precision"] = (TP+eps)/(TP+FP+eps)
        # if "precision" in metrics:
        #     metric_values["precision"] = (TP+eps)/(TP+FN+eps)
        metrics_avgs = {m: np.array(g).mean() for m, g in metric_values.items()}
        return metrics_avgs

    def plot_prediction(self,model, index=0, fig=None, figsize=(10,10), N=190, overlap=True, reshape_transform=None, modalities=None):

        # loading the image: it can be a numpy.ndarray or a Data/Batch object
        # image, mask = self.dataset.next_batch(1, shuffle=False) # selects an aleatory value from the dataset
        sample = self.dataset[N]
        is_graph_tensor = isinstance(sample, (Data, Batch))
        if is_graph_tensor:
            # this graph tensor
            image = sample
            mask = sample.y
            image['batch']=torch.zeros_like(sample.x).long()
        else:
            image, mask = sample[0], sample[1]
            image = image.reshape([1]+list(image.shape))

        input = torch.tensor(image).float() if self.to_tensor else image.clone()
        input = input.to(self.device)
        prediction = model(input)
        if is_graph_tensor:
            prediction =  prediction.x
        pred_mask = (sigmoid(prediction) > self.opt_th).float() if self.sigmoid else (prediction > self.opt_th).float()
        # after using prediction for calculating the mask then the prediction is transformed to prob,
        prediction = sigmoid(prediction) if self.sigmoid else prediction
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
            N = prediction.size

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
            C = N-A-B-C
            a = (A+C)/N
            p = (A + epsilon)/(A+B+epsilon)
            r = (A+epsilon)/(A+C+epsilon)
            dcm = 2*(p*r+epsilon)/(p+r+epsilon)
            print('OVERLAY IMAGE STATS ==> Accuracy: ', a ,' Precision: ', p, ', Recall: ', r, 'Dice: ', dcm)
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
        return fig

    def plot_volumen(self,model, index=0, fig=None, figsize=(10,10), N=1, overlap=True, reshape_transform=None, modalities=None):
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
        for sample in self.dataset.get_by_case_id(case_id):
            if is_graph_tensor:
                # this graph tensor
                image = sample
                mask = sample.y
                image['batch'] = torch.zeros_like(sample.x)
            else:
                image, mask = sample[0], sample[1]
                image = image.reshape([1]+list(image.shape))

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
                # concatenate the modality channels and prediction results
                pred_results = np.stack([mask, prediction, pred_mask], axis=-1)
                image = np.concatenate([image, pred_results], axis=-1)
                images.append(image)
        # the resulting image is Z,Y,X for overlap, and Z,Y,X,C for no overlap.
        result = np.stack(images).astype(np.float32)
        return result


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

