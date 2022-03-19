import torch 
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import *

def train(net, model_name, opt, loss, data_iter, epochs=EPOCHS):
    from time import time
    net.train()
    train_loss = []
    
    for epoch in range(epochs):
        t = 0
        for x, y in tqdm(data_iter):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pre = net(x)
            L = loss(pre, y)
            t += L.item() / x.shape[0]
            opt.zero_grad()
            L.backward()
            opt.step()
        print(f"epoch:{epoch} loss:{t}")
        train_loss.append(t)
    
    plt.plot(range(epochs), train_loss) 
    plt.show()
    torch.save(net.state_dict() ,model_name+"_"+str(time()))



























