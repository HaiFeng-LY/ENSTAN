import torch.nn as nn
import torch
from torch.autograd import Function
import torch.nn.functional as F
from typing import Optional, Any, Tuple
import numpy as np

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

class WarmStartGradientReverseLayer(nn.Module):
    def __init__(self, alpha= 1.0, lo= 0., hi= 1.,max_iters= 1000,auto_step=True):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, x):
        if self.auto_step:
            self.iter_num += 1
        coeff = float(self.iter_num) / self.max_iters
        coeff = 2. / (1. + np.exp(-10 * coeff)) - 1
        return GradientReverseFunction.apply(x, coeff * self.alpha)

def binary_accuracy(output, target):
     with torch.no_grad():
        pred = (output >= 0.5).float()
        correct = (pred == target).float().sum()
        accuracy = correct / target.size(0)
     return accuracy

class Discriminator(nn.Module):
    def __init__(self,hidden_1):
        super(Discriminator,self).__init__()
        self.fc1=nn.Linear(hidden_1,hidden_1)
        self.fc2=nn.Linear(hidden_1,1)
          
    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        x=torch.sigmoid(x)
        return x 
      
class DomainAdversarialLoss(nn.Module):
    def __init__(self,hidden_1, reduction: Optional[str] = 'mean',max_iter=1000):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0., hi=1., max_iters=max_iter, auto_step=True)
        self.domain_discriminator = Discriminator(hidden_1)
        self.bce = nn.BCELoss(reduction=reduction)
        self.domain_discriminator_accuracy = None
    def forward(self, f_s, f_t,alpha):
        self.grl.alpha = alpha
        f_s = self.grl(f_s)
        f_t = self.grl(f_t)
        d_s = self.domain_discriminator(f_s)
        d_t = self.domain_discriminator(f_t)
        d_label_s = torch.ones(d_s.size(0), 1).to(f_s.device)
        d_label_t = torch.zeros(d_t.size(0), 1).to(f_t.device)
        self.domain_discriminator_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))


        return 0.5 * (self.bce(d_s, d_label_s) + self.bce(d_t, d_label_t))