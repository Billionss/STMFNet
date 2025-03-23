import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from PIL import Image



class MyDataset(Dataset):
    
    def __init__(self, dataset, flag, window_size, horizon,  transform=None, image_transform=None):
        assert flag in ['train', 'val', 'test'], "Flag should be either 'train', 'val', or 'test'"
        
        if dataset =='Beijing':
            img_dir = './datasets/beijing/bj_jpg/'
            dataframe = pd.read_csv('./datasets/beijing/beijing.csv', encoding='gbk').iloc[:, 1:]   # 去除时间列
            img_paths = sorted([os.path.join(img_dir, img) for img in os.listdir(img_dir)])
        
        elif dataset == 'Tianjin':
            img_dir = './datasets/tianjin/tj_jpg/'
            dataframe = pd.read_csv('./datasets/tianjin/tianjin.csv', encoding='gbk').iloc[:, 1:] 
            img_paths = sorted([os.path.join(img_dir, img) for img in os.listdir(img_dir)])
        
        # 标准归一化 sklearn
        self.scaler = StandardScaler()
        dataframe = self.scaler.fit_transform(dataframe) # numpy

        # Calculate the sizes of the splits
        if dataset =='Beijing':
            train_size = int(0.7 * len(dataframe))
            val_size = int(0.1 * len(dataframe))
            test_size = len(dataframe) - train_size - val_size
        elif dataset == 'Tianjin':
            train_size = int(0.8 * len(dataframe))
            val_size = int(0.1 * len(dataframe))
            test_size = len(dataframe) - train_size - val_size

        # Split the dataframe into train, val, and test
        train_df = dataframe[:train_size]
        val_df = dataframe[train_size : train_size+val_size]
        test_df = dataframe[train_size+val_size:]

        # Split the img_paths into train, val, and test
        train_img_paths = img_paths[:len(train_df)//24]
        val_img_paths = img_paths[len(train_df)//24 : (len(train_df)+len(val_df))//24]
        test_img_paths = img_paths[(len(train_df)+len(val_df))//24:]

        if flag == 'train':
            self.dataframe = train_df
            self.img_paths = train_img_paths
        elif flag == 'val':
            self.dataframe = val_df
            self.img_paths = val_img_paths
        else:
            self.dataframe = test_df
            self.img_paths = test_img_paths

        print(self.dataframe.shape)
        self.window_size = window_size
        self.horizon = horizon
        self.img_num = window_size // 24
        self.img_label_num = horizon // 24
        self.transform = transform
        self.image_transform = image_transform

    def __len__(self):
        # 减去窗口大小和预测步长，确保每个样本都有对应的标签
        return len(self.dataframe) - self.window_size - self.horizon + 1

    def __getitem__(self, idx):
        # 获取窗口数据
        data = self.dataframe[idx : idx + self.window_size, :]
        # 获取标签，即窗口后horizon步的数据
        label = self.dataframe[idx + self.window_size  : idx + self.window_size + self.horizon, :]
        
        img_paths = self.img_paths[idx//24  : idx//24 + self.img_num]

        # img_paths = self.img_paths[idx*self.img_num : (idx+1)*self.img_num]
        images = [Image.open(img_path) for img_path in img_paths]

        image_label_paths = self.img_paths[(idx + self.window_size) // 24 : (idx + self.window_size) // 24 + self.img_label_num]
        # image_labels = [Image.open(image_label_path) for image_label_path in image_label_paths]

        if self.transform:
            data = self.transform(data)
            label = self.transform(label)
        if self.image_transform:
            images = [self.image_transform(image) for image in images]
            # image_labels = [self.image_transform(image_label) for image_label in image_labels]

        images = torch.stack(images)
        # image_labels = torch.stack(image_labels)
        return data, images, label
    
    def inverse_transform(self, data):
        # 对数据进行反归一化
        shape = data.shape
        data = data.reshape(-1, shape[-1])
        data = self.scaler.inverse_transform(data)
        data = data.reshape(shape)
        return data
