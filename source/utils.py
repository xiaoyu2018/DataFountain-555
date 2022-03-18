from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch

from config import DATA_DIR
import json
from PIL import Image
from os import path


# 标签与索引转换
# 支持合并转换和独立转换
def idx2name(*ids):
    period=["Dawn","Morning","Afternoon","Dusk","夜晚"]
    weather=["Cloudy","Sunny","Rainy","Snowy","Foggy"]
    
    if(len(ids)==1):
        return "".join([period[ids[0]//5],weather[ids[0]%5]]),None

    return period[ids[0]],weather[ids[1]]
def name2idx(*names):
    period={"Dawn":0,"Morning":1,"Afternoon":2,"黄昏":3,"夜晚":4}
    weather={"Cloudy":0,"Sunny":1,"Rainy":2,"雪天":3,"雾天":4}
    
    if(len(names)==1):
        return period[names[0][:2]]*5+weather[names[0][2:]],None
    return period[names[0]],weather[names[1]]




class ImgDataset(Dataset):
    
    def __init__(self,mode):
        super().__init__()
        self.mode=mode
        self.path=""
        self.data_info=[] #用于保存训练集或验证集标注信息，或测试集图片路径

        if(mode=="train" or mode=="val"):
            self.path+=DATA_DIR+f"\{mode}"
            anno_path=self.path+r"\annotations.json"
            

            with open(anno_path) as f:
                anno_data=json.load(f)
                self.data_info=anno_data["annotations"]
            
            # print(self.data_info)

        elif(mode=="test"):
            self.path+=r"\test"

        
        
        else:
            raise Exception("没有该模式的数据集")


        
    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        
        if(self.mode=="test"):
            return

        img=Image.open(path.join(self.path,self.data_info[index]["filename"]))
        y1,y2=name2idx(
            self.data_info["period"],
            self.data_info["weather"],
            )
        return transforms.ToTensor()(img),\
            (torch.LongTensor(y1),torch.LongTensor(y2)) \
                if y2 else torch.LongTensor(y1)







if __name__=='__main__':
    print(ImgDataset("train").__getitem__(2))

    
    