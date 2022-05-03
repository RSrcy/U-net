import torch
from torch.utils.data import Dataset
import os
from osgeo import gdal
import numpy as np
import cv2


class MyDataset(Dataset):  # 重写Dataset类
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.path = self.root_dir
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):  # 重写__getitem__方法，返回第idx张图像的tensor数组
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, img_name)
        img = gdal.Open(img_item_path)
        img_np = img.ReadAsArray()
        img_np_2 = img_np / 1.0
        img_tensor = torch.FloatTensor(img_np_2)

        a = img_name.split(".")
        new = a[0] + "_json/label.png"
        pathb = "E:/test2/json"
        label_path = pathb + '/' + new
        label = cv2.imread(label_path)
        label2 = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        label3 = torch.from_numpy(label2).permute(2, 0, 1)
        label_np = np.array(label3)
        label_tensor = torch.tensor(label_np)
        return img_tensor, label_tensor

    def __len__(self):  # 返回文件夹下训练集的总数量
        return len(self.img_path)
