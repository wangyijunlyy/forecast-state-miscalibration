import argparse
import torch
from model_ppm.exp.exp_main import Exp_Main
import random
import numpy as np
import setproctitle

if __name__ == '__main__':
    setproctitle.setproctitle('D3U_thread')

    parser = argparse.ArgumentParser(description='D3U-Framework-for-Probabilistic-MTS-Forecastingg')

    # basic config
    parser.add_argument('--is_training', type=int, default=1,  help='status')
    parser.add_argument('--model_id', type=str, default='ETTh2_96_192', help='model id')
    parser.add_argument('--model', type=str, default='PatchTST',
                        help='model name, options: [ns_Transformer, SVQ]')

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--data_name', type=str, default='custom', help='ETTh2')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh2.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    
    
    ##Model save path: checkpoints, pre-trained conditional network model save path: pretrain_checkpoints
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--pretrain_checkpoints', type=str, default='./pretrain_checkpoints/', help='location of condition model checkpoints')
    
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=192, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')

    # Model
    parser.add_argument('--d_model', type=int, default=256, help='dimension of condition_model')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of encoder layers')

    parser.add_argument('--d_model_f', type=int, default=256, help='dimension of condition_model')
    parser.add_argument('--d_ff_f', type=int, default=512, help='dimension of fcn')

    # other
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')  # 32
    parser.add_argument('--test_batch_size', type=int, default=8, help='batch size of train input data')  # 32
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate for diffusion model')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--cond_learning_rate', type=float, default=0.0001, help='optimizer learning rate for cond model')
    parser.add_argument('--std_learning_rate', type=float, default=0.0005, help='optimizer learning rate for std model')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=1, help='gpu')
    parser.add_argument('--use_multi_gpu', type=bool, default=False, help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')

    # KDE loss
    parser.add_argument('--K', type=int, default=4, help='')
    parser.add_argument('--tau', type=float, default=0.1, help='')  # soft 0.1 log 0.1
    parser.add_argument('--var', type=float, default=0.3, help='') # soft 1 log 0.3
    parser.add_argument('--nu', type=float, default=5, help='') # student-t

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.seed == -1:
        fix_seed = np.random.randint(2147483647)
    else:
        fix_seed = args.seed

    args.label_len = 0

    print('Using seed:', fix_seed)

    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    if args.use_gpu:
        if args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]
        else:
            torch.cuda.set_device(args.gpu)

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main
    

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_df{}_{}'.format(
                args.model_id,
                args.model,
                args.data_name,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.d_ff,
                args.des, ii)
            
            args.setting = setting

            exp = Exp(args)  # set experiments
            
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # exp.test_cond(setting)
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_df{}_{}'.format(
                args.model_id,
                args.model,
                args.data_name,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.d_ff,
                args.des, ii)
        args.setting = setting

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
