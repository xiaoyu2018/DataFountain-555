from operator import mod
import torch 
from torch import nn
from torch.nn import functional as F
import models
import utils
from tqdm import tqdm

model_name=""
net,model_name = models.get_resnet().to(torch.device('cuda:0'))

def train(epochs, opt, loss, data_iter):
    import matplotlib.pyplot as plt
    from time import time
    net.train()
    train_loss = []
    for epoch in range(epochs):
        t = 0
        for x, y in tqdm(data_iter):
            x = x.to(torch.device('cuda:0'))
            y = y.to(torch.device('cuda:0')).view(-1)
            pre = net(x)
            L = loss(pre, y)
            t += L.item() / x.shape[0]
            opt.zero_grad()
            L.backward()
            opt.step()
        train_loss.append(t)
    plt.plot(range(epochs), train_loss) 
    plt.show()
    torch.save(net.state_dict() ,model_name+str(time()))

train(epochs=10, 
      opt=torch.optim.Adam(net.parameters(), lr=0.03, weight_decay=0), 
      loss=torch.nn.CrossEntropyLoss(),
      data_iter = utils.load_data_12classes(batch_size=16, mode='val'))


























