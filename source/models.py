from torch import nn
from torch.nn import functional as F
from config import *


class Residual(nn.Module):
    def __init__(self, input_channels, output_channels, use_1x1conv=False, strides=1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, stride=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(output_channels) 
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.name = 'resnet'
    def forward(self, X):
        Y = F.leaky_relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.leaky_relu(Y)

#高宽减半
def resnet_block(input_channels, output_channels, num_residuals, first_block=True):
    blk = []
    for i in range(num_residuals):
        if i == 0:
            if first_block == True:
                blk.append(Residual(input_channels, output_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(input_channels, output_channels, use_1x1conv=True, strides=2)) 
        else:
            blk.append(Residual(output_channels, output_channels)) 
    return blk        
        

def get_resnet():
    l1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),#高宽不变
                   nn.BatchNorm2d(32), 
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1),#高宽大致减半
                   )        
    l2 = nn.Sequential(*resnet_block(32, 64, num_residuals=1))
    l3 = nn.Sequential(*resnet_block(64, 128, num_residuals=1, first_block=False))
    #l4 = nn.Sequential(*resnet_block(128, 256, num_residuals=1, first_block=False))
    #l5 = nn.Sequential(*resnet_block(256, 512, num_residuals=2, first_block=False))

    net = nn.Sequential(l1, l2, l3,  
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(128, 12))
    
    return net,'resnet'

'''        
X = torch.rand(size=(1, 3, 960, 960))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)        
'''        



def define_net(name:str,pretrained=False):
    from torchvision import models
    from torch import nn
    # ------------在此定义并返回所有网络------------
    def _1():
        net=models.resnet18(pretrained)
        net.fc=nn.Linear(net.fc.in_features,12)
        return net
    def _2():
        net=models.resnet34(pretrained)
        net.fc=nn.Linear(net.fc.in_features,12)
        return net
    def _3():
        net=models.mobilenet_v3_small(pretrained)
        net.classifier[-1]=nn.Linear(net.classifier[-1].in_features,12)
        return net
    def _4():
        net=models.mobilenet_v3_large(pretrained)
        net.classifier[-1]=nn.Linear(net.classifier[-1].in_features,12)
        return net
    def _defalut():
        raise Exception("未定义此类网络")

    ###################################################
    switch={
        'resnet18':_1,
        'resnet34':_2,
        'mobilenet_v3_small':_3,
        'mobilenet_v3_large':_4,
    }

    try:
        return switch.get(name,_defalut)()
    
    except Exception as e:
        print(e)

# ------------------预训练模型------------------ 
from torchvision import models

class PretrainedModel():

    def __init__(self,num_classes) -> None:
        self.name=""
        self.num_classes=num_classes
    
    def get_name(self):
        return self.name

    def _load_net(self):
        pass
    
    def get_model(self):
        net=self._load_net()
        net.fc=nn.Linear(net.fc.in_features,self.num_classes)          
        return net.to(DEVICE)

class Res18(PretrainedModel):

    def __init__(self,num_classes=12) -> None:
        super(Res18,self).__init__(num_classes)
        self.name="res18"
    
    def _load_net(self):
        return models.resnet18(False)
        
class Res50(PretrainedModel):
    def __init__(self,num_classes=12) -> None:
        super(Res50,self).__init__(num_classes)
        self.name="res50"
        
    def _load_net(self):
        return models.resnet50(False)
  
        
if __name__=='__main__':
    print(Res18().get_model())
        
        

