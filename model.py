import torch
import torch.nn as nn
import sinabs.layers as sl
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF

spike_grad = surrogate.fast_sigmoid(slope=75)
beta = 0.5

class Regression(nn.Module):
    """
    Simple Network with 2 convs and 1 fc
    """
    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(2, 8, 5),
            nn.MaxPool2d(2),
            # snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            sl.LIF(tau_mem=2, spike_fn=None),
            nn.Conv2d(8, 16, 5),
            nn.MaxPool2d(2),
            # snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            sl.LIF(tau_mem=2, spike_fn=None),
            nn.Flatten(),
            nn.Linear(13456,1024),
            nn.Linear(1024, 2),
            # snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
            sl.LIF(tau_mem=2, spike_fn=None),
        )


class SmartDoorClassifierv1(nn.Module):
    """ 
    The initial smartdoor code without dropout
    """
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            # 2 x 128 x 128
            # Core 0
            nn.Conv2d(2, 8, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=False),  # 8, 64, 64
            sl.LIFSqueeze(tau_mem=5e-3, batch_size=16),
            # nn.AvgPool2d(kernel_size=(2, 2)),  # 8,32,32
            nn.MaxPool2d(2),
            # """Core 1"""
            nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # 16, 32, 32
            sl.LIFSqueeze(tau_mem=5e-3, batch_size=16),
            # nn.AvgPool2d(kernel_size=(2, 2)),  # 16, 16, 16
            nn.MaxPool2d(2),
            # """Core 2"""
            nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # 8, 16, 16
            sl.LIFSqueeze(tau_mem=5e-3, batch_size=16),
            # nn.AvgPool2d(kernel_size=(2, 2)),  # 16, 16, 16
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, 2, bias=False),
            sl.LIFSqueeze(tau_mem=5e-3, batch_size=16, spike_fn=None),
        )
    def forward(self, x):
        return self.seq(x)