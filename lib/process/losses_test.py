from unittest import TestCase
import torch
from .losses import DCS, DiceLoss
from .evaluation import dice_coeff
import matplotlib.pyplot as plt


from torch import sigmoid


class DCS_mixed(object):
    def __init__(self, pre_sigmoid=False):
        self.pre_sigmoid = pre_sigmoid
    def __call__(self, inputs, targets):

        """
        This definition generalize to real valued pred and target vector.
        This should be differentiable.
        pred: tensor with first dimension as batch
        target: tensor with first dimension as batch
        """

        smooth = 1.
        # inputs.dims() = (B, H,W,1)---> (B, H*W)
        # have to use contiguous since they may from a torch.view op
        iflat = inputs.contiguous().view(-1) if not self.pre_sigmoid else sigmoid(inputs.contiguous().view(-1))
        tflat = targets.contiguous().view(-1)
        intersection = (iflat*tflat).sum()

        A_sum = torch.sum(iflat*iflat)
        B_sum = torch.sum(tflat*tflat)

        return 1-((2.*intersection+smooth)/(A_sum+B_sum+smooth))


def modified_loss(_pred, _target, pre_sigmoid=True, smooth=1.):
    iflat = _pred.flatten(start_dim=1)
    tflat = _target.flatten(start_dim=1)
    _pred = torch.sigmoid(_pred) if pre_sigmoid else _pred
    intersection = (iflat * tflat).sum(axis=1)  # (B,1)
    A_sum = torch.sum(iflat * iflat, axis=1)
    B_sum = torch.sum(tflat * tflat, axis=1)
    loss_mod = torch.mean(1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth)))
    return loss_mod

class TestDiceLoss(TestCase):

    def setUp(self) -> None:
        self.a = torch.tensor([[[0.1,0.1], [0.9,0.9]],[[0.1, 0.1], [0.9, 0.9]]])
        self.b = torch.tensor([[[1.0,0.0], [1.0,1.0]],[[1.0, 0.0], [1.0, 1.0]]])

    def test_forward(self):
        print(self.a.shape)
        dice_loss = DiceLoss(pre_sigmoid=True, epsilon=1)
        a_logit = torch.tensor([[[-200.0,-200.0], [1000,1000]],[[-200.0, -200.0], [1000, 1000]]])
        loss = dice_loss(a_logit, self.b)
        print('loss = ', loss)
        # without sigmoid
        dice_loss = DiceLoss(pre_sigmoid=False, epsilon=1)
        loss = dice_loss(self.a, self.b)
        print('loss = ', loss)

    def test_backward_DCS(self):
        weights = torch.tensor([[1.0, 1.0], [1.0, 1.0]], requires_grad=True)
        pre_sigmoid = False
        loss = DiceLoss(pre_sigmoid=pre_sigmoid)
        learning_rate = 1.0
        losses_mixed = []
        losses_mod = []

        print('training for 100 iterations')
        for i in range(1000):
            pred = torch.stack([self.a[0] * weights, self.a[1] * weights], axis=0)
            L = loss(pred, self.b)
            print("loss values: ", L)
            external_grads = torch.tensor(1)
            L.backward(external_grads)
            # print('gradientes=> ', weights.grad)
            with torch.no_grad():
                weights -= learning_rate * weights.grad
                weights.grad.zero_()
                losses_mixed.append(L.cpu().detach().numpy())
        # modification:
        print('original: pred', torch.sigmoid(pred), " \n  target=", self.b)
        weights = torch.tensor([[1.0, 1.0], [1.0, 1.0]], requires_grad=True)
        loss_mod = DCS(pre_sigmoid=pre_sigmoid)

        print('training for 100 iterations modified version')
        for i in range(1000):
            pred = torch.stack([self.a[0] * weights, self.a[1] * weights], axis=0)
            L = loss_mod(pred, self.b)
            print("loss values: ", L)
            external_grads = torch.tensor(1)
            L.backward(external_grads)
            # print('gradientes=> ', weights.grad)
            with torch.no_grad():
                weights -= learning_rate * weights.grad
                weights.grad.zero_()
                losses_mod.append(L.cpu().detach().numpy())
        print('modified: pred', torch.sigmoid(pred), " \n  target=", self.b)

        print('plotting:  ')
        plt.plot(losses_mod, label="Dice loss smooth")
        plt.plot(losses_mixed, label="Dice loss V2")
        plt.xlabel('iterations')
        plt.ylabel('dice loss')
        plt.legend()
        plt.show()


class TestDCS(TestCase):
    def setUp(self) -> None:
        self.a = torch.tensor([[[0.1,0.1], [0.9,0.9]],[[0.1, 0.1], [0.9, 0.9]]])
        self.b = torch.tensor([[[1.0,0.0], [1.0,1.0]],[[1.0, 0.0], [1.0, 1.0]]])

    def test_DCS(self):
        loss = DCS_mixed()
        loss_mixed = loss(self.a,self.b)
        metric = dice_coeff(self.a, self.b)
        print('metric : ', metric)
        self.assertAlmostEqual(metric.numpy(), 0.76, delta=0.0001)
        loss = DCS()
        loss_mod = loss(self.a, self.b)
        print('modified loss: ', loss_mod)
        self.assertNotEqual(loss_mixed, loss_mod)

    def test_backward_DCS(self):
        weights = torch.tensor([[1.0,1.0], [1.0,1.0]], requires_grad=True)
        pre_sigmoid = True
        loss = DCS_mixed(pre_sigmoid=pre_sigmoid)
        learning_rate = 1.0
        losses_mixed = []
        losses_mod = []

        print('training for 100 iterations')
        for i in range(1000):
            pred = torch.stack([self.a[0]*weights, self.a[1]*weights], axis=0)
            L = loss(pred, self.b)
            print("loss values: " , L)
            external_grads = torch.tensor(1)
            L.backward(external_grads)
            # print('gradientes=> ', weights.grad)
            with torch.no_grad():
                weights -= learning_rate * weights.grad
                weights.grad.zero_()
                losses_mixed.append(L.cpu().detach().numpy())
        # modification:
        print('original: pred', torch.sigmoid(pred), " \n  target=", self.b)
        weights = torch.tensor([[1.0,1.0], [1.0,1.0]], requires_grad=True)
        loss_mod = DCS(pre_sigmoid=pre_sigmoid)

        print('training for 100 iterations modified version')
        for i in range(1000):
            pred = torch.stack([self.a[0] * weights, self.a[1] * weights], axis=0)
            L = loss_mod(pred, self.b)
            print("loss values: ", L)
            external_grads = torch.tensor(1)
            L.backward(external_grads)
            # print('gradientes=> ', weights.grad)
            with torch.no_grad():
                weights -= learning_rate* weights.grad
                weights.grad.zero_()
                losses_mod.append(L.cpu().detach().numpy())
        print('modified: pred', torch.sigmoid(pred), " \n  target=", self.b)

        print('plotting:  ' )
        plt.plot(losses_mod)
        plt.plot(losses_mixed)
        plt.xlabel('iterations')
        plt.ylabel('dice loss')
        plt.show()

        self.assertGreater(losses_mixed[-1], losses_mod[-1])



