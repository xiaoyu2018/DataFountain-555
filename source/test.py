import torch
from utils import load_data, idx2name


def gen_submission(net, param_path=None, merge=True):
    import json
    net.eval()
    
    if(param_path):
        net.load_state_dict(torch.load(param_path)) 

    test_iter = load_data(batch_size=4, mode='test', merge=merge)
    predicts = []
    with torch.no_grad():
        for x, y in test_iter:
            x = x.to(torch.device('cuda:0'))
            pre = net(x)
            labels = [idx2name(i)[0] for i in torch.argmax(pre,1)]
            
            for i in range(len(labels)):
                d = dict()
                d['filename'] = 'test_images\\'+y[i]
                d['period'], d['weather'] = tuple(labels[i].split(' '))
                predicts.append(d)         
    
    res = dict()
    res["annotations"] = predicts
    with open("submission.json","w") as f:
        json.dump(res, f)

