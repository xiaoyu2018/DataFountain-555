import torch
from utils import load_data, idx2name,listdir
from torchvision import transforms
from config import *


def inference_all(net,param_path,pretrained=True,merge=True):
    net.eval()
    net=net.to(DEVICE)

    if(param_path):
        net.load_state_dict(torch.load(param_path)) 

    test_iter = load_data(batch_size=8, mode='test', merge=merge)
    predicts = []
    
    with torch.no_grad():
        for x in test_iter:
            x = x.to(torch.device(DEVICE))
            x=transforms.Resize(IMG_SIZE)(x)
            if(pretrained):
                x=transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x)
            out = net(x)
            
            predicts.extend(torch.argmax(out,1).cpu().numpy())
    
    # 完整测试集的预测结果，(400)
    return predicts


def get_voting_nets(models_path:str=BASE_DIR+r"\voting_models"):
    """_summary_

    Args:
        models_path (str): 存储所有投票模型参数的文件夹路径
    Returns:
        返回(net,param_path) 组成的list
    """
    from models import define_net

    types=listdir(models_path)
    models_info=[]
    
    for type in types:
        spec_net_dir=models_path+"\\"+type
        params_path=listdir(spec_net_dir)

        for p in params_path:
            p=spec_net_dir+"\\"+p
            models_info.append((define_net(type),p))

    return models_info
def vote_to_inference_all(models_info:list,merge=True):
    """_summary_

    Args:
        models_info (list): (net,param_path) 组成的list
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

    res=torch.argmax(scores_map,dim=1)
    return res


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
    from models import define_net
    
    # param=r"D:\code\machine_learning\DataFountain-555\params\mobilenet_v3_small_768_15.pt"
    
    # net1=models.mobilenet_v3_small(pretrained=True,progress=True)
    # net1.classifier[-1]=nn.Linear(net1.classifier[-1].in_features,12)
    # net2=models.mobilenet_v3_small(pretrained=True,progress=True)
    # net2.classifier[-1]=nn.Linear(net2.classifier[-1].in_features,12)
    
    # info=[(net1,param),(net2,param)]

    # print(vote_to_inference_all(info))
    
    # predicts = inference_all(net,param)

    # gen_submission(predicts)

    # print(define_voting_net('resnet18'))

    models_info=get_voting_nets()
    predicts=vote_to_inference_all(models_info)
    # predicts=inference_all(define_net('resnet18'),r'D:\code\machine_learning\DataFountain-555\voting_models\resnet18\resnet_20.pt')
    gen_submission(predicts)
    