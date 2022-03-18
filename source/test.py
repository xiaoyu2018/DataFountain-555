import torch
from utils import load_data_12classes, idx2name






def gen_submission(net, merge=True):
    import json
    net.load_state_dict(torch.load(r'C:\Users\SXQ\Desktop\Ai_stu\resnet1647615352.2196805')) 
    test_iter = load_data_12classes(batch_size=4, mode='test', merge=True)
    list = []
    for x, y in test_iter:
        x = x.to(torch.device('cuda:0'))
        pre = net(x)
        labels = [idx2name(i)[0] for i in torch.argmax(pre,1)]
        for i in range(len(labels)):
            d = dict()
            d['filename'] = 'test_images\\'+y[i]
            d['period'], d['weather'] = tuple(labels[i].split(' '))
            list.append(d)         
    d = dict()
    d["annotations"] = list
    with open("submission.json","w") as f:
        json.dump(d, f)

