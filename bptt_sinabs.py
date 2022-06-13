import torch
import torch.nn as nn
import torchvision
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import tonic.transforms as transforms
import tonic
from torch.utils.data import DataLoader
from tonic import DiskCachedDataset
import sinabs.layers as sl 
from aermanager.datasets import SpikeTrainDataset
import numpy as np
import os
import math
from torchsummary import summary

sensor_size = tonic.datasets.NMNIST.sensor_size

# Denoise removes isolated, one-off events
# time_window
frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000), 
                                      transforms.ToFrame(sensor_size=sensor_size, 
                                                         time_window=1000)
                                     ])

# trainset = tonic.datasets.NMNIST(save_to='/home/jyz/ssd/snn/data2', transform=frame_transform, target_transform=int, train=True)
# testset = tonic.datasets.NMNIST(save_to='/home/jyz/ssd/snn/data2', transform=frame_transform, target_transform=int, train=False)

trainset = SpikeTrainDataset(
    source_folder='/home/jyz/ssd/Speck2b/event_0522_2000_new/', 
    transform=np.float32,
    # target_transform=int,
    dt=1000
)

transform = tonic.transforms.Compose([torch.from_numpy,
                                      torchvision.transforms.RandomRotation([-10,10])])

# cached_trainset = DiskCachedDataset(trainset, transform=transform, target_transform=int, cache_path='/home/jyz/ssd/snn/cache/nmnist/train')

# no augmentations for the testset
# cached_testset = DiskCachedDataset(testset, target_transform=int, cache_path='/home/jyz/ssd/snn/cache/nmnist/test')

batch_size = 16
trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), drop_last=True, shuffle=False)
# testloader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# neuron and simulation parameters
spike_grad = surrogate.fast_sigmoid(slope=75)
beta = 0.5

#  Initialize Network
# net = nn.Sequential(nn.Conv2d(2, 8, 5),
#                     nn.MaxPool2d(2),
#                     # snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
#                     sl.LIF(tau_mem=2, spike_fn=None),
#                     nn.Conv2d(8, 16, 5),
#                     nn.MaxPool2d(2),
#                     # snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
#                     sl.LIF(tau_mem=2, spike_fn=None),
#                     nn.Flatten(),
#                     nn.Linear(13456,1024),
#                     nn.Linear(1024, 2),
#                     # snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
#                     sl.LIF(tau_mem=2, spike_fn=None),
#                     ).to(device)

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

net = SmartDoorClassifierv1().to(device)

# print(summary(net, (2,128,128)))

def forward_pass(net, data, batch_size):  
    spk_rec = []
    mem_rec = []
    if isinstance(net, nn.Module):
        # utils.reset(net)  # resets hidden states for all LIF neurons in net
        for layer in net.seq:
            if isinstance(layer, sl.LIF):
                layer.reset_states()

        time_steps, batch_size, *shape = data.shape
        data = data.permute(1,0,2,3,4)
        data = data.reshape((-1, 2, 128, 128))
        spk_out = net(data)
        spk_out = spk_out.reshape(batch_size, -1, 2)
        # reset_state()
        # for step in range(data.size(0)):  # data.size(0) = number of time steps
        #     spk_out = net(data[step])
        #     # spk_out = net[8].v_mem
        #     spk_rec.append(spk_out)
            # mem_rec.append(mem_out)
    else:
        net.reset_state()
        for step in range(data.size(0)):  # data.size(0) = number of time steps
            spk_out = net(data[step])
            spk_rec.append(spk_out)
  
    return spk_out

optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))

# loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
loss_fn = nn.MSELoss()

num_epochs = 40

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 35, 0.1)

num_iters = 20

loss_hist = []
acc_hist = []
total = 0
all_distance = 0
# training loop
for epoch in range(num_epochs):
    total = 0
    for i, (data, targets) in enumerate(trainloader):
        # model.reset_states()
        data = data.to(device)
        targets = targets.to(device) / 128
        # data = data.reshape((-1, 2, 34, 34))
        net.train()
        # spk_rec = net(data)
        spk_rec = forward_pass(net, data, batch_size=16)
        spk_rec = torch.mean(spk_rec, dim=1)
        loss_val = loss_fn(spk_rec, targets[:,0].float())
    
        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())
        total += loss_val.item()
        # This will end training after 50 iterations by default
        
        if (i + 1) % num_iters == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Learning_rate: %.5f' %(epoch + 1, num_epochs, i + 1, len(trainset) // batch_size, total / (i + 1), scheduler.get_last_lr()[0]))
    scheduler.step()
    # total = 0
    # all_distance = 0
    # with torch.no_grad():
    #     net.eval()
    #     for batch, (img, label) in enumerate(testloader):
    #         img = img.to(device)
    #         label = label.to(device)

    #         outputs = forward_pass(net, data)
    #         outputs = torch.mean(spk_rec, dim=0)

    #         distance = pow(outputs.to("cpu") *128 - (targets * 128), 2).sum(1)
    #         total += targets.size(0)
    #         all_distance += distance.sum()
        
    #     ave_dis = float(all_distance) / float(total)
    #     print('Total test samples: %d .' % total)
    #     print('=========  Test ave_dis = %.5f . ==========' % math.sqrt(ave_dis))

    if (epoch + 1) % 4 == 0:
        print("saving model...")

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(net, 'checkpoint/bptt_' + str(epoch+1) + '.pth')
    

