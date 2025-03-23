from data_provider.data_factory import data_provider
from .exp_basic import Exp_Basic
from model import vt, DLinear, t, RSTS, TS, Timesnet, RSV
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import torch
import torch.nn as nn
from torch import optim, autograd

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'DLinear': DLinear,
            'RSTS' :RSTS,
            'TS': TS,
            'RSV': RSV,
        }

        model = model_dict[self.args.model].Model(self.args).float()

        return model

    #flag = 'train' or 'val' or 'test'
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):

        # ts_block_params = list(self.model.seasonal_block.parameters()) + list(self.model.trend_block.parameters())
        # ts_block_ids = {id(p): True for p in ts_block_params}
        #
        # rs_block_params = list(self.model.rs_block.parameters())
        # rs_block_ids = {id(p): True for p in rs_block_params}
        # #
        # fusion_params =  list(self.model.fusion.parameters())
        # fusion_block_ids = {id(p): True for p in fusion_params}
        #
        # excluded_ids = ts_block_ids.keys() | rs_block_ids.keys() | fusion_block_ids.keys()
        #
        # other_params = [p for p in self.model.parameters() if id(p) not in excluded_ids]
        #
        # print(len(ts_block_params), len(rs_block_params), len(fusion_params), len(other_params))
        #
        # model_optim = optim.Adam([
        #     {'params': ts_block_params},
        #     {'params': rs_block_params,'lr': 0.5*self.args.learning_rate},
        #     {'params': fusion_params,'lr': 0.5*self.args.learning_rate},
        #     {'params': other_params },
        # ], lr=self.args.learning_rate)

        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.L1Loss()
        # criterion = nn.MSELoss()
        return criterion


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_ts, batch_rs, batch_y) in enumerate(vali_loader):

                batch_ts = batch_ts.float().to(self.device)
                batch_rs = batch_rs.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # decoder input
                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder

                outputs = self.model(batch_ts, batch_rs)


                pred = torch.from_numpy(vali_data.inverse_transform(outputs.detach().cpu().numpy()))
                true = torch.from_numpy(vali_data.inverse_transform(batch_y.detach().cpu().numpy()))

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        #use automatic mixed precision training
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_ts, batch_rs, ts_label) in enumerate(train_loader):

                iter_count += 1
                model_optim.zero_grad()

                batch_ts = batch_ts.float().to(self.device)
                batch_rs = batch_rs.float().to(self.device)
                ts_label = ts_label.float().to(self.device)

                outputs = self.model(batch_ts, batch_rs)

                # predict = torch.from_numpy(train_data.inverse_transform(outputs.cpu().numpy()))
                # target = torch.from_numpy(train_data.inverse_transform(batch_y.cpu().numpy()))

                # print(outputs.shape,batch_y.shape)

                loss = criterion(outputs, ts_label)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()


                with autograd.detect_anomaly():
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        graph_list = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_ts, batch_rs, batch_y) in enumerate(test_loader):

                batch_ts = batch_ts.float().to(self.device)
                batch_rs = batch_rs.float().to(self.device)
                batch_y = batch_y.float().to(self.device)


                outputs = self.model(batch_ts, batch_rs)

                # graph_list.append(graph)


                outputs = torch.from_numpy(test_data.inverse_transform(outputs.detach().cpu().numpy()))  # outputs.detach().cpu().numpy()
                batch_y = torch.from_numpy(test_data.inverse_transform(batch_y.detach().cpu().numpy()))

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)  # .detach().cpu().numpy()
                trues.append(true) # .detach().cpu().numpy()
                # inputx.append(batch_ts.detach().cpu().numpy())
                if i % 10 == 0:
                    input = batch_ts.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        #See utils / tools for usage
        # if self.args.test_flop:
        #     test_params_flop((batch_x.shape[1],batch_x.shape[2]))
        #     exit()
        # print('preds_shape:', len(preds),len(preds[0]),len(preds[1]))

        preds = torch.cat(preds,dim=0)
        trues = torch.cat(trues,dim=0)
        # preds = np.array(preds)
        # trues = np.array(trues)
        # inputx = np.array(inputx)

        print('preds_shape:', preds.shape)
        print('trues_shape:', trues.shape)

        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        preds = preds.cpu().numpy()
        trues = trues.cpu().numpy()

        mae, mse, rmse, mape, mspe, rse, corr, nd, nrmse, r2 = metric(preds, trues)
        print('mae:{}, rmse:{}, mape:{}, r2:{}'.format(mae, rmse,mape ,r2))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('nd:{}, nrmse:{}, mse:{}, mae:{}, rse:{}, mape:{}'.format(nd, nrmse,mse, mae, rse, mape))
        f.write('\n')
        f.write('\n')
        f.close()


        print(os.path.join(folder_path, 'real_prediction.npy'))

        # pred_dict = {'pred': preds, 'true': trues}
        # torch.save(pred_dict, folder_path + 'pred_results.pth')
        # os.path.join(folder_path, 'real_prediction.npy')

        # np.save(os.path.join(folder_path, 'real_prediction.npy'), preds)
        # np.save(os.path.join(folder_path, 'real_target.npy'), trues)


        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # np.save(folder_path + 'real_prediction.npy', preds)

        return
