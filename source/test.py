import torch
from utils import load_data, idx2name,listdir
from torchvision import transforms
from config import *


def inference_all(net,param_path,pretrained=True,merge=True):
    net.eval()
    net=net.cuda()

    if(param_path):
        net.load_state_dict(torch.load(param_path)) 

    test_iter = load_data(batch_size=40, mode='test', merge=merge)
    predicts = []
    
    with torch.no_grad():
        for x in test_iter:
            x = x.to(torch.device(DEVICE))
            if(pretrained):
                x=transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x)
            out = net(x)
            
            predicts.extend(torch.argmax(out,1).cpu().numpy())
    
    # 完整测试集的预测结果，(400)
    return predicts


def define_voting_net(type:str):
    from torchvision import models
    switch={
        'resnet18':_1,
        'resnet34':_2,
        'mobilenet_v3_small':_3,
        'mobilenet_v3_large':_4,
    }

    try:
        return switch.get(type,_defalut)()
    
    except Exception as e:
        print(e)


    # ------------在此定义并返回所有网络------------
    def _1():
        net=models.resnet18()
        net.fc=nn.Linear(net.fc.in_features,12)
        return net
    def _2():
        net=models.resnet34()
        net.fc=nn.Linear(net.fc.in_features,12)
        return net
    def _3():
        net=models.mobilenet_v3_small(pretrained=True,progress=True)
        net.classifier[-1]=nn.Linear(net.classifier[-1].in_features,12)
    def _4():
        net=models.mobilenet_v3_large()
        net.classifier[-1]=nn.Linear(net.classifier[-1].in_features,12)
        return net
    
    
    def _defalut():
        raise Exception("未定义此类网络")

def vote_to_inference_all(models_info:list,merge=True):
    """_summary_

    Args:
        models_info (list): (netparam_path) 组成的list
        merge (bool, optional): 是否合并时间与天气

    Returns:
        索引形式返回全部测试集的预测结果
    """
    scores_map=torch.zeros((400,12 if merge else 7))
    
    for net,param_path in models_info:
        predicts=inference_all(net,param_path)
        predicts=[range(400),predicts] # [第0维全部想要操作的索引，第1维全部想要操作的索引]
        print(predicts)
        scores_map[predicts]+=1

    predicts=torch.argmax(scores_map,dim=1)
    return predicts


def gen_submission(predicts, merge=True):
    import json

    img_path=DATA_DIR+r"\test"+r"\test_images"
    filenames=listdir(img_path)

    labels = [idx2name(i)[0] for i in predicts]
    gen_info=[]
    
    for i in range(len(labels)):
        d = dict()
        d['filename'] = 'test_images\\'+filenames[i]
        d['period'], d['weather'] = tuple(labels[i].split(' '))
        gen_info.append(d)         
    
    res = dict()
    res["annotations"] = gen_info
    with open("submission.json","w") as f:
        json.dump(res, f)


if __name__ == '__main__':
    from torchvision import models
    from torch import nn
    
    param=r"D:\code\machine_learning\DataFountain-555\params\mobilenet_v3_small_768_15.pt"
    
    net1=models.mobilenet_v3_small(pretrained=True,progress=True)
    net1.classifier[-1]=nn.Linear(net1.classifier[-1].in_features,12)
    net2=models.mobilenet_v3_small(pretrained=True,progress=True)
    net2.classifier[-1]=nn.Linear(net2.classifier[-1].in_features,12)
    
    info=[(net1,param),(net2,param)]

    print(vote_to_inference_all(info))
    
    # predicts = inference_all(net,param)

    # gen_submission(predicts)