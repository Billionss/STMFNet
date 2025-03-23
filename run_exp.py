import argparse
import os
import time
from multiprocessing import freeze_support
import torch
from exp.exp_main import Exp_Main
from exp.exp_formers import Exp_Formers
import random
import numpy as np

fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='This is description')

# basic config
parser.add_argument('--model', type=str, required=False, default='RSTS', help='model dict,see details in exp_main')
parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, mask, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
parser.add_argument('--visual_graph', type=bool, default=True, help='')

# data loader
parser.add_argument('--data', type=str, required=False, default='Tianjin', help='dataset type')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
parser.add_argument('--label_len', type=int, default=1, help='pred length')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# model configs
parser.add_argument('--top_k', type=int, default=5, help='for FFT period')
parser.add_argument('--num_nodes', type=int, default=27, help='num of stations, Beijing: 34, Tianjin:27')
parser.add_argument('--num_channels', type=int, default=1, help='channels of rs image data')
parser.add_argument('--embed_dim', type=int, default=64, help='embedding dim ')
parser.add_argument('--ts_layers', type=int, default=3, help='layer nums of timeseries block')

# DLinear
parser.add_argument('--pred_len', type=int, default=72, help='pred length')
parser.add_argument('--individual', type=bool, default=False, help='individual')
parser.add_argument('--enc_in', type=int, default=34, help='encoder input size')

# Timesnet
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')

# Formers Tansformer  Autoformer
# model define
parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')


parser.add_argument('--moving_avg', default=13, help='window size of moving average')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')

parser.add_argument('--dec_in', type=int, default=34, help='decoder input size')
parser.add_argument('--c_out', type=int, default=34, help='output size')

parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--embed_type', type=int, default=0, help='prediction sequence length')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                         'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')

# optimization
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MAE', help='loss function')
parser.add_argument('--lradj', type=str, default='3', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

print('Args in experiment:')
print(args)

Exp = Exp_Main
# Exp = Exp_Formers

if args.is_training:
    start = time.time()
    for ii in range(args.itr):
        setting = '{}_{}_sl{}_ll{}_ed{}'.format(
            args.model,
            args.data,
            args.seq_len,
            args.label_len,
            args.embed_dim)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training >>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.train(setting)

        print('>>>>>>>testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting)

        torch.cuda.empty_cache()
    end = time.time()
    used_time = end -start
    print("time:",used_time)
    f = open("result.txt", 'a')
    f.write('time:{}'.format(used_time))
    f.write('\n')
    f.write('\n')
    f.close()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                  args.model,
                                                                                                  args.data,
                                                                                                  args.features,
                                                                                                  args.seq_len,
                                                                                                  args.label_len,
                                                                                                  args.pred_len,
                                                                                                  args.d_model,
                                                                                                  args.n_heads,
                                                                                                  args.e_layers,
                                                                                                  args.d_layers,
                                                                                                  args.d_ff,
                                                                                                  args.factor,
                                                                                                  args.embed,
                                                                                                  args.distil,
                                                                                                  args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()

