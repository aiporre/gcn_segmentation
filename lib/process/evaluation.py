import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch_geometric.data import Data, Batch

from .losses import DiceCoeff
from .progress_bar import printProgressBar
from torch import sigmoid
import numpy as np

from ..graph.batch import to_torch_batch


class Evaluator(object):
    def __init__(self, dataset, batch_size=64, to_tensor=True, device=None, sigmoid=False):
        self.dataset = dataset.test
        self._batch_size = batch_size
        self.dataset.enforce_batch(self._batch_size)
        self.to_tensor = to_tensor
        self.device = device if device is not None else torch.device('cpu')
        self.sigmoid = sigmoid


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
            pred_mask = (sigmoid(prediction) > 0.5).float() if self.sigmoid else (prediction > 0.5).float()
            # reorganize prediction according to the batch.
            if not pred_mask.size(0) == label.size(0):
                b = label.size(0)
                pred_mask = pred_mask.view(b, -1)
            DCM_accum.append(dice_coeff(pred_mask, label).item())
            N += label.numel()
            if progress_bar:
                printProgressBar(i, L, prefix='DCM:', suffix='Complete', length=50)
            else:
                print('DCS Epoch: in batch ', i+1, ' out of ', L, '(percentage {}%)'.format(100.0*(i+1)/L))
            i += 1

        # self.dataset.enforce_batch(1)

        return np.array(DCM_accum).mean()

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
            pred = (sigmoid(prediction) > 0.5).long() if self.sigmoid else (prediction > 0.5).long()
            if not pred.size(0) == label.size(0):
                b = label.size(0)
                pred = pred.view(b, -1)
            if len(label.shape)>2:
                b = label.size(0)
                label = label.view(b, -1)
                pred = pred.view(b, -1)
            correct += pred.eq(label).sum().item()
            N += label.numel()
            mask_pos = label.eq(1).squeeze().nonzero()
            mask_neg = label.eq(0).squeeze().nonzero()
            TP += pred[:,mask_pos].eq(label[:,mask_pos]).sum().item()
            FP += pred[:,mask_pos].ne(label[:,mask_pos]).sum().item()
            FN += pred[:,mask_neg].ne(label[:,mask_neg]).sum().item()

            if progress_bar:
                printProgressBar(i, L, prefix='Acc, Rec, Pre:', suffix='Complete', length=50)
            else:
                print('Bin Scores: in batch ', i+1, ' out of ', L, '(Completed {}%)'.format(100.0*(i+1)/L))
            i += 1
        return correct/N, TP/(TP+FP+eps), TP/(TP+FN+eps)

    def plot_prediction(self,model, index=0, fig=None, figsize=(10,10), N=190, overlap=True, reshape_transform=None):

        # loading the image: it can be a numpy.ndarray or a Data/Batch object
        # image, mask = self.dataset.next_batch(1, shuffle=False) # selects an aleatory value from the dataset
        sample = self.dataset[N]
        if not isinstance(sample,tuple):
            # this graph tensor
            image = sample
            mask = sample.y
            image['batch']=torch.zeros_like(sample.x)
        else:
            image, mask = sample[0], sample[1]
            image = image.reshape([1]+list(image.shape))

        input = torch.tensor(image).float() if self.to_tensor else image.clone()
        input = input.to(self.device)
        prediction = model(input)
        # pred_mask = (sigmoid(prediction) > 0.5).float()
        if isinstance(prediction, (Data, Batch)):
            prediction =  prediction.x
        pred_mask = (sigmoid(prediction) > 0.5).float() if self.sigmoid else (prediction > 0.5).float()

        if not isinstance(image,np.ndarray):
            dimension = image.x.size(0)# it will assume a square image, if the transform is None.
            if reshape_transform is None:
                dimension = np.sqrt(dimension).astype(int)
                mask = mask.cpu().detach().numpy().reshape((dimension,dimension))
                image = image.x.cpu().detach().numpy().reshape((dimension, dimension))
                prediction = prediction.reshape((dimension, dimension))
                pred_mask = pred_mask.reshape((dimension,dimension))
            else:
                mask = reshape_transform(mask.cpu().detach().numpy())
                image = reshape_transform(image.x.cpu().detach().numpy())
                prediction = reshape_transform(torch.sigmoid(prediction))
                pred_mask = reshape_transform(pred_mask)

        # plot input image
        if not fig:
            fig = plt.figure(figsize=figsize)
        if overlap:
            pred_mask = pred_mask.squeeze()
            cmap_TP = ListedColormap([[73/255, 213/255, 125/255, 1]])
            cmap_FP = ListedColormap([[255/255, 101/255, 80/255, 1]])
            cmap_FN = ListedColormap([[15/255, 71/255, 196/255, 1]])
            TP = pred_mask.cpu().numpy()*mask
            FP = 1*((pred_mask.cpu().numpy()-mask) > 0)
            FN = 1*((mask-pred_mask.cpu().numpy()) > 0)
            N = prediction.numel()

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
            print('Accuracy: ', a ,' Precision: ', p, ', Recall: ', r, 'Dice: ', dcm)
        else:
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3)
            ax4 = fig.add_subplot(2, 2, 4)
            ax1.imshow(image.copy().squeeze(),cmap='gray')
            ax1.set_title('original image')
            # plot p(y=1|X=x)
            ax2.imshow(prediction.cpu().detach().numpy().squeeze(), cmap='gray')
            ax2.set_title('probability map')
            # plot mask
            ax3.imshow(mask.squeeze(),cmap='gray')
            ax3.set_title('ground truth mask')
            # plot prediction
            ax4.imshow(pred_mask.cpu().detach().numpy().squeeze(),cmap='gray')
            ax4.set_title('predicted mask >0.5 prob')
        return fig

    def plot_volumen(self,model, index=0, fig=None, figsize=(10,10), N=190, overlap=True, reshape_transform=None):


        images = []
        for ii in range(index,N):
            sample = self.dataset[ii]
            if not isinstance(sample, tuple):
                # this graph tensor
                image = sample
                mask = sample.y
                image['batch'] = torch.zeros_like(sample.x)
            else:
                image, mask = sample[0], sample[1]
                image = image.reshape([1]+list(image.shape))

            input = torch.tensor(image).float() if self.to_tensor else image.clone()
            input = input.to(self.device)
            prediction = model(input)
            # pred_mask = (sigmoid(prediction) > 0.5).float()
            if isinstance(prediction, (Data, Batch)):
                prediction = prediction.x
            pred_mask = (sigmoid(prediction) > 0.5).float() if self.sigmoid else (prediction > 0.5).float()

            if not isinstance(image, np.ndarray):
                dimension = image.x.size(0)  # it will assume a square image,if the transform is None.t :
                if reshape_transform is None:
                    dimension = np.sqrt(dimension).astype(int)
                    mask = mask.cpu().detach().numpy().reshape((dimension,dimension))
                    image = image.x.cpu().detach().numpy().reshape((dimension, dimension))
                    prediction = prediction.reshape((dimension, dimension))
                    pred_mask = pred_mask.reshape((dimension,dimension))
                else:
                    mask = reshape_transform(mask.cpu().detach().numpy())
                    image = reshape_transform(image.x.cpu().detach().numpy())
                    prediction = reshape_transform(torch.sigmoid(prediction))
                    pred_mask = reshape_transform(pred_mask)
            TP = pred_mask.cpu().numpy()*mask
            FP = 1*((pred_mask.cpu().numpy()-mask) > 0)
            FN = 1*((mask-pred_mask.cpu().numpy()) > 0)
            mix = TP+2*FP+3*FN
            images.append(mix.squeeze())
        result = np.stack(images).astype('float32')
        return result
        # # plot input image
        # # TODO: image will change its shape I need a transformer class
        # if not fig:
        #     fig = plt.figure(figsize=figsize)
        # if overlap:
        #     pred_mask = pred_mask.squeeze()
        #     cmap_TP = ListedColormap([[73/255, 213/255, 125/255, 1]])
        #     cmap_FP = ListedColormap([[255/255, 101/255, 80/255, 1]])
        #     cmap_FN = ListedColormap([[15/255, 71/255, 196/255, 1]])
        #     TP = pred_mask.cpu().numpy()*mask
        #     FP = 1*((pred_mask.cpu().numpy()-mask) > 0)
        #     FN = 1*((mask-pred_mask.cpu().numpy()) > 0)
        #     N = prediction.numel()
        #
        #     alpha = 0.5
        #     fig = plt.figure(figsize=(10, 10))
        #     ax = fig.add_subplot(1, 1, 1)
        #     ax.imshow(image.copy().squeeze(), cmap='gray')
        #     masked = np.ma.masked_where(FP == 0, FP)
        #     ax.imshow(masked, cmap=cmap_FP, alpha=alpha)
        #     masked = np.ma.masked_where(FN == 0, FN)
        #     ax.imshow(masked, cmap=cmap_FN, alpha=alpha)
        #     masked = np.ma.masked_where(TP == 0, TP)
        #     ax.imshow(masked, cmap=cmap_TP, alpha=alpha)
        #
        #     A = TP.sum()
        #     B = FP.sum()
        #     C = FN.sum()
        #     C = N-A-B-C
        #     a = (A+C)/N
        #     p = A/(A+B)
        #     r = A/(A+C)
        #     dcm = 2*p*r/(p+r)
        #     print('Accuracy: ', a, ' Precision: ', p, ', Recall: ', r, 'Dice: ', dcm)
        # else:
        #     ax1 = fig.add_subplot(2, 2, 1)
        #     ax2 = fig.add_subplot(2, 2, 2)
        #     ax3 = fig.add_subplot(2, 2, 3)
        #     ax4 = fig.add_subplot(2, 2, 4)
        #     ax1.imshow(image.copy().squeeze(), cmap='gray')
        #     ax1.set_title('original image')
        #     # plot p(y=1|X=x)
        #     ax2.imshow(prediction.cpu().detach().numpy().squeeze(), cmap='gray')
        #     ax2.set_title('probability map')
        #     # plot mask
        #     ax3.imshow(mask.squeeze(), cmap='gray')
        #     ax3.set_title('ground truth mask')
        #     # plot prediction
        #     ax4.imshow(pred_mask.cpu().detach().numpy().squeeze(), cmap='gray')
        #     ax4.set_title('predicted mask >0.5 prob')
        # return fig

class KEvaluator(Evaluator):
    def __init__(self, dataset, batch_size=64, to_tensor=True, device=None, sigmoid=False):
        '''
        Keras addaptation evaluate model
        :param dataset:
        :param batch_size:
        :param to_tensor:
        :param device:
        :param sigmoid:
        '''
        super(KEvaluator, self).__init__(dataset, batch_size=batch_size, to_tensor=to_tensor, device=device, sigmoid=sigmoid)
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
            pred_mask = (pred_mask[:,1,:,:] > 0.5).float()

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
                    print('DCS Epoch: in batch ', i+1, ' out of ', L, '(percentage {}%)'.format(100.0*(i+1)/L))
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
        pred_mask = (prediction > 0.5)

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

