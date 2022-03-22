import models
import utils
from train  import train
from test import gen_submission
import torch 
import models

from config import *


if __name__=='__main__':
      # net,model_name = models.get_resnet()
      # net = net.to(torch.device('cuda:0'))

      # train(net, model_name, epochs=15, 
      #       opt=torch.optim.Adam(net.parameters(), lr=0.03, weight_decay=0.5), 
      #       loss=torch.nn.CrossEntropyLoss(),
      #       data_iter = utils.load_data(batch_size=8, mode='val'))

      # gen_submission(net)

      # res_18=models.Res18()
      # name,net=res_18.get_name(),res_18.get_model()
      from torchvision import models
      from torch import nn
      # convn_t=models.convnext_tiny(pretrained=True,progress=True)
      # convn_t.classifier[2]=nn.Linear(convn_t.classifier[2].in_features,12)
      # convn_t=convn_t.to(DEVICE)

      # params_1x=[]
      # params_1x.extend(
      #       [param for param in convn_t.features.parameters()]+
      #       [param for param in convn_t.classifier[0].parameters()]
      # ) 

      
      # resnet18=models.resnet18(pretrained=True,progress=True)
      # resnet18.fc=nn.Linear(resnet18.fc.in_features,12)
      # resnet18=resnet18.cuda()
      # params_1x = [param for name, param in resnet18.named_parameters()
      #        if name not in ["fc.weight", "fc.bias"]]

      net=models.mobilenet_v3_small(pretrained=True,progress=True)
      net.classifier[-1]=nn.Linear(net.classifier[-1].in_features,12)
      net=net.cuda()
      # params_1x = net.features.parameters()
      # train(net, "mobilenet_v3_large_768", epochs=30, 
      #       opt=torch.optim.Adam([{'params': params_1x},
      #                              {'params': net.classifier.parameters(),
      #                               'lr': 5e-4 * 10}],lr=5e-4, weight_decay=1e-3), 
      #       loss=torch.nn.CrossEntropyLoss(),
      #       data_iter = utils.load_data(batch_size=4, mode='train'))
      
      gen_submission(net,PARAMS_DIR+"/mobilenet_v3_small_768_15.pt")
