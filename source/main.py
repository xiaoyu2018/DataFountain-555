import models
import utils
from train  import train
import test
import torch 
import models


net,model_name = models.get_resnet()
net = net.to(torch.device('cuda:0'))

train(net, model_name, epochs=15, 
      opt=torch.optim.Adam(net.parameters(), lr=0.03, weight_decay=0.5), 
      loss=torch.nn.CrossEntropyLoss(),
      data_iter = utils.load_data_12classes(batch_size=8, mode='val'))

test.gen_submission(net)
