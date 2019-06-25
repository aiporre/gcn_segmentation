import torch
from torch.autograd import Function, Variable
import matplotlib.pyplot as plt
from .progress_bar import printProgressBar
from lib.utils import print_debug
import numpy as np
class Evaluator(object):
    def __init__(self, dataset, batch_size=64, to_tensor=True, device=None):
        self.dataset = dataset.test
        self._batch_size = batch_size
        self.dataset.enforce_batch(self._batch_size)
        self.to_tensor = to_tensor
        self.device = device if device is not None else torch.device('cpu')


    def DCM(self, model, progress_bar=True):
        DCM_accum = 0
        N = len(self.dataset)
        L = self.dataset.num_batches
        if progress_bar:
            printProgressBar(0, L, prefix='DCM:', suffix='Complete', length=50)
        i = 0
        for image, label in self.dataset.batches():
            features = torch.tensor(image).float() if self.to_tensor else image
            label = torch.tensor(label).float() if self.to_tensor else label
            features = features.to(self.device)
            label = label.to(self.device)
            prediction = model(features)
            pred_mask = (prediction > 0.5).float()
            # reorganize prediction according to the batch.
            if not pred_mask.size(0) == label.size(0):
                b = label.size(0)
                pred_mask = pred_mask.view(b, -1)
            DCM_accum += dice_coeff(pred_mask, label).item()
            i += 1
            if progress_bar:
                printProgressBar(i, L, prefix='DCM:', suffix='Complete', length=50)
            else:
                print('Training Epoch: in batch ', i+1, ' out of ', L, '(percentage {}%)'.format(100.0*(i+1)/L))
        self.dataset.enforce_batch(1)

        return DCM_accum/N

    def plot_prediction(self,model, index=0, fig=None, figsize=(10,10)):
        if not fig:
            fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)
        # loading the image: it can be a numpy.ndarray or a Data/Batch object
        image, mask = self.dataset.next_batch(1) # selects an aleatory value from the dataset

        input = torch.tensor(image).float() if self.to_tensor else image.clone()
        input = input.to(self.device)
        prediction = model(input)
        pred_mask = (prediction > 0.5).float()

        if not image is np.ndarray:
            dimension = image.x.size(0)# it will assume a square image, though we need a transformer for that
            dimension = np.sqrt(dimension).astype(int)
            image = image.x.cpu().detach().numpy().reshape((dimension, dimension))
            mask = mask.cpu().detach().numpy().reshape((dimension,dimension))
            pred_mask = pred_mask.reshape((dimension,dimension))


        # plot input image
        #TODO: image will change its shape I need a transformer class

        ax1.imshow(image.copy().squeeze())
        # plot mask
        ax2.imshow(mask.squeeze())
        # plot prediction
        ax3.imshow(pred_mask.cpu().detach().numpy().squeeze())
        return fig






class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        eps = 0.0001
        try:
            self.inter = torch.dot(inputs.view(-1), targets.view(-1))
        except RuntimeError as e:
            message = 'inputs'+str(inputs.size())+'targets'+str(targets.size())
            print_debug(message)
            print_debug('Error calculation in intersection', exception=e)
            raise e

        self.union = torch.sum(inputs)+torch.sum(targets)+eps

        t = (2*self.inter.float()+eps)/self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_tensors
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output*2*(target*self.union-self.inter) \
                         /(self.union*self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

def dice_coeff(inputs, target):
    """Dice coeff for batches"""
    if inputs.is_cuda:
        s = torch.tensor(1).float().cuda().zero_()
    else:
        s = torch.tensor(1).float().zero_()

    for i, c in enumerate(zip(inputs, target)):
        s = s+DiceCoeff().forward(c[0], c[1])

    return s/(i+1)