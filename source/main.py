import models
import utils
from train  import train
from test import gen_submission
import torch 
import models




if __name__=='__main__':
      # net,model_name = models.get_resnet()
      # net = net.to(torch.device('cuda:0'))

      # train(net, model_name, epochs=15, 
      #       opt=torch.optim.Adam(net.parameters(), lr=0.03, weight_decay=0.5), 
      #       loss=torch.nn.CrossEntropyLoss(),
      #       data_iter = utils.load_data(batch_size=8, mode='val'))

      # gen_submission(net)

      res_18=models.Res18()
      name,net=res_18.get_name(),res_18.get_model()

      # train(net, name, epochs=30, 
      #       opt=torch.optim.Adam(net.parameters(), lr=0.03), 
      #       loss=torch.nn.CrossEntropyLoss(),
      #       data_iter = utils.load_data(batch_size=12, mode='train'))
      
      gen_submission(net,r"D:\code\machine_learning\DataFountain-555\source\res181647665821.281966")
