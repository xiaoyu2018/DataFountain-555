from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch

from config import DATA_DIR
import json
from PIL import Image
from os import path,listdir


# 标签与索引转换
# 支持合并转换和独立转换
def idx2name(*ids):
    period=["Dawn","Morning","Afternoon","Dusk"]
    weather=["Cloudy","Sunny","Rainy"]
    
    if(len(ids)==1):
        return " ".join([period[ids[0]//3],weather[ids[0]%3]]),None

    return period[ids[0]],weather[ids[1]]
def name2idx(*names):
    period={"Dawn":0,"Morning":1,"Afternoon":2,"Dusk":3}
    weather={"Cloudy":0,"Sunny":1,"Rainy":2}
    
    if(len(names)==1):
        p,w=names[0].split(" ")
        return period[p]*3+weather[w],None
    return period[names[0]],weather[names[1]]




class ImgDataset(Dataset):
    
    def __init__(self,mode,merge=False,resize=None):
        super().__init__()
        self.resize=resize
        self.merge=merge
        self.mode=mode
        self.path=DATA_DIR
        self.data_info=[] #用于保存训练集或验证集标注信息，或测试集图片路径

        if(mode=="train" or mode=="val"):
            self.path+=f"\{mode}"
            anno_path=self.path+r"\annotations.json"
            

            with open(anno_path) as f:
                anno_data=json.load(f)
                self.data_info=anno_data["annotations"]
            
            # print(self.data_info)

        elif(mode=="test"):
            self.path+=r"\test"
            img_path=self.path+r"\test_images"
            self.data_info=listdir(img_path)
            
        
        
        else:
            raise Exception("没有该模式的数据集")


        
    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        
        if(self.mode=="test"):
            file_name=self.data_info[index]
            img=Image.open(path.join(self.path+r"\test_images",file_name))
            
            if(self.resize):
                img=transforms.Resize(self.resize)(img)
            # 返回图片张量，图片名
            return transforms.ToTensor()(img),file_name




        img=Image.open(path.join(self.path,self.data_info[index]["filename"]))
        p=self.data_info[index]["period"]
        w=self.data_info[index]["weather"]
        
        if(self.merge):
            y1,y2=name2idx(p+" "+w)
        else:
            y1,y2=name2idx(p,w)

        if(self.resize):
            img=transforms.Resize(self.resize)(img)
        # 返回图片张量，标签
        return transforms.ToTensor()(img),\
            (torch.LongTensor([y1]),torch.LongTensor([y2])) \
                if y2!=None else torch.LongTensor([y1])







if __name__=='__main__':
    print(ImgDataset("test",resize=(720,720)).__getitem__(1)[0].shape)
    print(ImgDataset("train",merge=True).__getitem__(1)[0].shape)

    
    

    
    