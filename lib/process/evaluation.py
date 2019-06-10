import torch
from torch.autograd import Function, Variable
import matplotlib.pyplot as plt
class Evaluator(object):
    def __init__(self, dataset):
        self.dataset = dataset.test
    def DCM(self, model):
        DCM_accum = 0
        N = len(self.dataset)
        # TODO: Do it in batches!!
        for image, label in self.dataset:
            features = torch.tensor(image).unsqueeze(0).float()
            label = torch.tensor(label).unsqueeze(0).float()
            prediction = model(features)
            pred_mask = (prediction > 0.5).float()
            DCM_accum += dice_coeff(pred_mask, label).item()
        return DCM_accum/N

    def plot_prediction(self,model, index=0, fig=None, figsize=(10,10)):
        if not fig:
            fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)

        image, mask = self.dataset[index]
        # plot input image
        #TODO: image will change its shape I need a transformer class
        ax1.imshow(image.copy().squeeze())
        # plot mask
        ax2.imshow(mask)
        # plot prediction
        input = torch.tensor(image).unsqueeze(0).float()
        prediction = model(input)
        pred_mask = (prediction > 0.5).float()
        ax3.imshow(pred_mask.detach().numpy().squeeze())
        return fig






class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        eps = 0.0001
        self.inter = torch.dot(inputs.view(-1), targets.view(-1))
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