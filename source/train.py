import torch 
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import *

def train(net, model_name, opt, loss, data_iter, epochs=EPOCHS):
    net.train()
    train_loss = []
    
    for epoch in range(epochs):
        t = []
        for x, y in tqdm(data_iter):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pre = net(x)
            L = loss(pre, y)
            t += L.item()
            opt.zero_grad()
            L.backward()
            opt.step()
        print(f"epoch:{epoch} loss:{sum(t)/len(t)}")
        train_loss.append(sum(t)/len(t))
        
        torch.save(net.state_dict() ,PARAMS_DIR+"/"+model_name+"_"+str(epoch)+".pt")


    plt.plot(range(epochs), train_loss) 
    plt.show()



























