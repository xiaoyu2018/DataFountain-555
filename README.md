<div id="top"></div>


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/xiaoyu2018/DataFountain-555">
    <img src="https://www.datafountain.cn/favicon.ico" width="50" height="50">
  </a>

  <h3 align="center"><font size="6">DataFountain赛题-555</font></h3>

  <p align="center">
    DataFountain 训练赛道赛题<a href="https://www.datafountain.cn/competitions/555"> 天气以及时间分类 </a> 
    <br />
    <br />
  </p>
</div>


<!-- 修改url -->
<div align="center">
  <a href="https://github.com/xiaoyu2018/DataFountain-555/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/xiaoyu2018/DataFountain-555.svg?style=for-the-badge">
  </a>  
  <a href="https://github.com/xiaoyu2018/DataFountain-555/stargazers">
    <img src="https://img.shields.io/github/stars/xiaoyu2018/DataFountain-555.svg?style=for-the-badge">
  </a>  
  <a href="https://github.com/xiaoyu2018/DataFountain-555/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/xiaoyu2018/DataFountain-555.svg?style=for-the-badge">
  </a>  
</div>
<br />

---



<!-- ABOUT THE PROJECT -->
## About The Project
本仓库记录了DataFountain赛题 天气以及时间分类 的解决方案。截至2022年3月27日，测试成绩为0.94934795，第二赛段排行榜A榜排名第一。  
<a href="https://www.datafountain.cn/competitions/555/ranking?isRedance=0&sch=1897">
    <img src="./score.png">
</a>

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started
1. 项目结构：  
    + DataFountain-555  
    --dataset（数据集）  
    --params（模型参数）  
    --source（源代码）  
    ----main.py  
    ----models.py  
    ----train.py   
    ----test.py   
    ----config.py   
    ----utils.py  
    --voting_models（用于投票的模型）  
    ----model_name1 （第一种模型）  
    ------1.pt  
    ------2.pt  
    ----model_name2 （第二种模型）  
    ------1.pt  
    ------2.pt  

2. 其他说明：  
    + 本项目将去除了训练集中未出现的三种类别，将剩余的`period`中["Dawn","Morning","Afternoon","Dusk"]与`weather`中["Cloudy","Sunny","Rainy"]合并为3x4=12类进行分类。
    + 进行投票的模型只参与Inference过程，且具有相同权重。
    + 所有经过预训练的模型均在ImageNet上完成预训练。
    + 将参与投票的模型参数并按网络结构进行分类，放至voting_models不同文件夹下。然后于`models.py`中`define_net`建立映射关系。

<p align="right">(<a href="#top">back to top</a>)</p>

## Methods
+ 迁移学习
+ 数据增强
+ 多模型投票
+ 去除训练集中未出现的类别

<p align="right">(<a href="#top">back to top</a>)</p>


