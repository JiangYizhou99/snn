from curses.ascii import SP
import torch
import torch.nn as nn
import tonic
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
from torch.utils.data import DataLoader
import sinabs.layers as sl 
from aermanager.datasets import SpikeTrainDataset
import numpy as np
import os
import math
from torchsummary import summary
from model import Regression, SmartDoorClassifierv1


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Get spike datasets
trainset = SpikeTrainDataset(
    source_folder='/home/jyz/ssd/Speck2b/event_0522_2000_new_split/Train/', 
    transform=np.float32,
    # target_transform=int,
    dt=1000
)

testset = SpikeTrainDataset(
    source_folder="/home/jyz/ssd/Speck2b/event_0522_2000_new_split/Test/",
    transform=np.float32,
    dt=1000
)

# For datasets framed by spike count, collate_fn should be used
batch_size = 16

trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), drop_last=True, shuffle=False)
testloader = DataLoader(testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), drop_last=True, shuffle=False)


# Network initialization
net = SmartDoorClassifierv1().to(device)

# define forward function for bptt
def forward_pass(net, data, batch_size):  

    if isinstance(net, nn.Module):
        # utils.reset(net)  # resets hidden states for all LIF neurons in net
        for layer in net.seq:
            if isinstance(layer, sl.LIF):
                layer.reset_states()
        time_steps, batch_size, *shape = data.shape
        data = data.permute(1,0,2,3,4)
        data = data.reshape((-1, 2, 128, 128))
        mem_out = net(data)
        mem_out = mem_out.reshape(batch_size, -1, 2)
       
    return mem_out


# Define optimizer and loss function
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
    total_loss = 0
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
        total_loss += loss_val.item()
        
        if (i + 1) % num_iters == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Learning_rate: %.5f' %(epoch + 1, num_epochs, i + 1, len(trainset) // batch_size, total_loss / (i + 1), scheduler.get_last_lr()[0]))
    scheduler.step()

    # Test result
    total = 0
    all_distance = 0
    with torch.no_grad():
        net.eval()
        for batch, (img, label) in enumerate(testloader):
            img = img.to(device)
            label = label.to(device)

            mem_out = forward_pass(net, data)
            outputs = torch.mean(mem_out, dim=1)

            distance = pow(outputs *128 - label, 2).sum(1)
            total += label.size(0)
            all_distance += distance.sum()
        
        ave_dis = float(all_distance) / float(total)
        print('Total test samples: %d .' % total)
        print('=========  Test ave_dis = %.5f . ==========' % math.sqrt(ave_dis))

    # Save model
    if (epoch + 1) % 4 == 0:
        print("saving model...")

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(net, 'checkpoint/bptt_' + str(epoch+1) + '.pth')
    

