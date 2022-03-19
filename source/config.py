# ------------公用模块------------
import torch


# ------------路径设置------------
BASE_DIR = r"D:\code\machine_learning\DataFountain-555"
DATA_DIR=BASE_DIR+r"\dataset"
PARAMS_DIR=BASE_DIR+"/params"

# ------------模型设置------------
BATCH_SIZE=16
EPOCHS=15
IMG_SIZE=(768,768)
DEVICE=torch.device("cuda:0")