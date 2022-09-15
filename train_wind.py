import argparse
import os
import torch
import wandb
import numpy as np
from torch.utils.tensorboard import SummaryWriter
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from experiments.exp_wind import Exp_wind

parser = argparse.ArgumentParser(description='ResNet on Wind Speed dataset')

parser.add_argument('--model', type=str, required=False, default='ResNet1D', help='model of the experiment')
### -------  dataset settings --------------
parser.add_argument('--data', type=str, required=False, default='Wind_speed_data', help='name of dataset')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='Wind_speed_data', help='location of the data file')
parser.add_argument('--features', type=str, default='S', choices=['S', 'M'], help='features S is univariate, M is multivariate')
parser.add_argument('--target', type=str, default='60m', help='target feature')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='exp/wind_checkpoints/', help='location of model checkpoints')
parser.add_argument('--inverse', type=bool, default =False, help='denorm the output data')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')


### -------  device settings --------------
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0',help='device ids of multile gpus')
                                                                                  
### -------  input/output length settings --------------                                                                            
parser.add_argument('--seq_len', type=int, default=32, help='input sequence length of resnet')
parser.add_argument('--label_len', type=int, default=32, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=64, help='prediction sequence length, horizon')
parser.add_argument('--concat_len', type=int, default=0)
parser.add_argument('--single_step', type=int, default=0)
parser.add_argument('--single_step_output_One', type=int, default=0)
parser.add_argument('--lastWeight', type=float, default=1.0)
                                                              
### -------  training settings --------------  
parser.add_argument('--cols', type=str, nargs='+', help='file list')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=0, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='RMSE',help='loss function')
parser.add_argument('--lradj', type=int, default=1,help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--save', type=bool, default =False, help='save the output results')
parser.add_argument('--model_name', type=str, default='ResNet1D')
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--evaluate', type=bool, default=False)

### -------  model settings --------------  

parser.add_argument('--input_channel', type=int, default=32, help='dim of input, the same as n_channel')
parser.add_argument('--stride', type=int, default=2, help='stride of kernel moving')
parser.add_argument('--base_filter', type=int, default=4, help='number of filters in the first several Conv layer, it will double at every 4 layers')
parser.add_argument('--groups', default=1, type=int, help='set largest to 1')
parser.add_argument('--n_block', default=4, type=int, help='number of blocks')
parser.add_argument('--kernel', default=3, type=int, help='width of kernel')
parser.add_argument('--n_classes', default=64, type=int, help='number of classes')



args = parser.parse_args()


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'Wind_speed_data': {'data': 'Wind_speed_data.csv', 'T': args.target, 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

wandb.init(config=args)
config = wandb.config

'''config = {
    'method': 'grid', #grid, random
    'parameters':{
    
      'train_epochs': {
          'values': [10]
      },
      'batch_size': {
          'values': [128]
      },
      'lr': {
          'values': [1e-4]
      },
      'features': {
          'values': ['S']
      },
      'input_channel': {
          'values': [32]
      },
      'n_classes': {
          'values': [64]
      }
    }
}

sweep_id = wandb.sweep(
    config,
    project="Wind Speed Forecasting")
'''
torch.manual_seed(4321)  # reproducible
torch.cuda.manual_seed_all(4321)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
torch.backends.cudnn.enabled = True

Exp = Exp_wind

mae_ = []
maes_ = []
mse_ = []
mses_ = []

if args.evaluate:
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_bs{}_itr0'.format(args.model,args.data, args.features, args.seq_len, args.label_len, args.pred_len,args.lr,args.batch_size)
    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    mae, maes, mse, mses = exp.test(setting, evaluate=True)
    print('Final mean normed mse:{:.4f},mae:{:.4f},denormed mse:{:.4f},mae:{:.4f}'.format(mse, mae, mses, maes))
else:
    if args.itr:
        for ii in args.itr:
            # setting record of experiments
            setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_bs{}_itr{}'.format(args.model,args.data, args.features, args.seq_len, args.label_len, args.pred_len,args.lr,args.batch_size,ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting) 

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            mae, maes, mse, mses = exp.test(setting)
            mae_.append(mae)
            mse_.append(mse)
            maes_.append(maes)
            mses_.append(mses)

            torch.cuda.empty_cache()
        print('Final mean normed mse:{:.4f}, std mse:{:.4f}, mae:{:.4f}, std mae:{:.4f}'.format(np.mean(mse_), np.std(mse_), np.mean(mae_),np.std(mae_)))
        print('Final mean denormed mse:{:.4f}, std mse:{:.4f}, mae:{:.4f}, std mae:{:.4f}'.format(np.mean(mses_),np.std(mses_), np.mean(maes_), np.std(maes_)))
        print('Final min normed mse:{:.4f}, mae:{:.4f}'.format(min(mse_), min(mae_)))
        print('Final min denormed mse:{:.4f}, mae:{:.4f}'.format(min(mses_), min(maes_)))
    else:
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_bs{}_itr0'.format(args.model,args.data, args.features, args.seq_len, args.label_len, args.pred_len,args.lr,args.batch_size)
        wandb.log({
            "model": args.model,
            "data": args.data,
            "features" : args.features,
            "Sequence Length": args.seq_len,
            "label length": args.label_len,
            "prediction length": args.pred_len,
            "learning rate": args.lr,
            "batch size": args.batch_size
            }) 
        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        #wandb.agent(sweep_id, exp.train(setting))

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        mae, maes, mse, mses =exp.test(setting)
        print('Final mean normed mse:{:.4f},mae:{:.4f},denormed mse:{:.4f},mae:{:.4f}'.format(mse, mae, mses, maes))



