import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, dataset, window_size, horizon, flag='train',  features='M', scale=True, timeenc=0, freq='h', transform=None, image_transform=None ):
        # size [seq_len, label_len, pred_len]
        # info

        self.seq_len = window_size
        self.label_len = horizon
        self.pred_len = horizon
        # init

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.dataset = dataset

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        if self.dataset =='Beijing':
            # img_dir = './datasets/beijing/bj_jpg/'
            dataframe = pd.read_csv('./datasets/beijing/beijing.csv', encoding='gbk')# .iloc[:, 1:]   # 去除时间列
            # img_paths = sorted([os.path.join(img_dir, img) for img in os.listdir(img_dir)])

        elif self.dataset == 'Tianjin':
            # img_dir = './datasets/tianjin/tj_jpg/'
            dataframe = pd.read_csv('./datasets/tianjin/tianjin.csv', encoding='gbk')# .iloc[:, 1:]
             # img_paths = sorted([os.path.join(img_dir, img) for img in os.listdir(img_dir)])

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(dataframe.columns)

        cols.remove('datetime')
        df_raw = dataframe[['datetime'] + cols  ]
        # print(cols)
        if self.dataset =='Beijing':
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
        elif self.dataset == 'Tianjin':
            num_train = int(len(df_raw) * 0.8)
            num_test = int(len(df_raw) * 0.1)
            num_vali = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]


        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['datetime']][border1:border2]
        df_stamp['datetime'] = pd.to_datetime(df_stamp.datetime)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.datetime.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.datetime.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.datetime.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.datetime.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['datetime'], axis=1 ).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['datetime'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    # def __getitem__(self, index):
    #     s_begin = index
    #     s_end = s_begin + self.seq_len
    #     r_begin = s_end - self.label_len
    #     r_end = r_begin + self.label_len + self.pred_len
    #
    #     seq_x = self.data_x[s_begin:s_end]
    #     seq_y = self.data_y[r_begin:r_end]
    #     seq_x_mark = self.data_stamp[s_begin:s_end]
    #     seq_y_mark = self.data_stamp[r_begin:r_end]
    #
    #     return seq_x, seq_y, seq_x_mark, seq_y_mark
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.label_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        # 对数据进行反归一化
        shape = data.shape
        data = data.reshape(-1, shape[-1])
        data = self.scaler.inverse_transform(data)
        data = data.reshape(shape)
        return data

