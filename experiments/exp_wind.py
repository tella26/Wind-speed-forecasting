import json
import os
import sys
import time

import numpy as np
import wandb
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')
from data_processing.wind_speed_dataloader import Dataset_wind, Dataset_Pred
from experiments.exp_basic import Exp_Basic
from data_processing.tools import EarlyStopping, adjust_learning_rate, save_model, load_model
from metrics.wind_speed_metrics import metric
from models.resnet1d import ResNet1D


class Exp_wind(Exp_Basic):
    def __init__(self, args):
        super(Exp_wind, self).__init__(args)
    
    def _build_model(self):

        if self.args.features == 'S':
            in_dim = 1
        elif self.args.features == 'M':
            in_dim = 7
        else:
            print('Error!')
            
        model = ResNet1D(
                in_channels= self.args.input_channel, 
                base_filters= self.args.base_filter, 
                kernel_size= self.args.kernel, 
                stride= self.args.stride, 
                groups= self.args.groups,
                n_block= self.args.n_block, 
                n_classes= self.args.n_classes, 
                downsample_gap=2, 
                increasefilter_gap=4, 
                use_bn=True, 
                use_do=True, 
                verbose=False     
                )
        writer = SummaryWriter('event/run_wind_speed_models/{}'.format(self.args.model_name))
        example_data = torch.randn(32, 32, 72).to(self.device)
        writer.add_graph(model, example_data)
        print(model)
        return model.double()

    def _get_data(self, flag):
        args = self.args
        Data =  Dataset_wind
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        dataset_json_str = json.dumps(list(data_set))
        wandb.log({
            flag + "_Dataset": dataset_json_str})
        print(flag, len(data_set))
        
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr)
        wandb.log({
            "Optimizer": model_optim })
        return model_optim
    
    def _select_criterion(self, losstype):
        if losstype == "mse":
            criterion = nn.MSELoss()
        elif losstype == "mae":
            criterion = nn.L1Loss()
        else:
            criterion = nn.L1Loss()
        return criterion

    def valid(self, valid_data, valid_loader, criterion):
        self.model.eval()
        total_loss = []

        preds = []
        trues = []
        mids = []
        pred_scales = []
        true_scales = []
        mid_scales = []
        #writer = SummaryWriter('event/run_wind_speed_metrics/{}'.format(self.args.model_name))

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(valid_loader):
            pred, pred_scale, mid, mid_scale, true, true_scale= self._process_one_batch_resnet1D(
                valid_data, batch_x, batch_y)
            true = true.mean(-1)
            true_scale = true_scale.mean(-1)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            pred_scale = torch.tensor(pred_scale)
            pred_scales.append(pred_scale.detach().cpu().numpy())
            true_scale = torch.tensor(true_scale)           
            true_scales.append(true_scale.detach().cpu().numpy())
            total_loss.append(loss)
            wandb.log({"Predicted Scales val": pred_scales})
            wandb.log({"True Scales val": true_scales})
            wandb.log({"Predicted val": pred})
            wandb.log({"True val": true})
            
        total_loss = np.average(total_loss)
        wandb.log({"Total validation loss": total_loss})

        preds = np.array(preds)
        trues = np.array(trues)
        pred_scales = np.array(pred_scales)
        true_scales = np.array(true_scales)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        true_scales = true_scales.reshape(-1, true_scales.shape[-2], true_scales.shape[-1])
        pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])

        mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
        maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)
        # For true and Predicted
        wandb.log({"mae val": mae})
        wandb.log({"mse val": mse})
        wandb.log({"rmse val": rmse})
        wandb.log({"mape val": mape})
        wandb.log({"mspe val": mspe})
        wandb.log({"corr val": corr})
        
         # For true and Predicted scales
        wandb.log({"mae_scale val": maes})
        wandb.log({"mse_scale val": mses})
        wandb.log({"rmse_scale val": rmses})
        wandb.log({"mape_scale val": mapes})
        wandb.log({"mspe_scale val": mspes})
        wandb.log({"corr_scale val": corrs})
       
        
        '''
        writer.add_scalar('mae', mae, global_step=self.epoch)
        writer.add_scalar('mse', mse, global_step=self.epoch)
        writer.add_scalar('rmse', rmse, global_step=self.epoch)
        writer.add_scalar('mspe', mspe, global_step=self.epoch)
        writer.add_scalar('corr', corr, global_step=self.epoch)
        '''
        print('final --> normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
        print('final --> denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mses, maes, rmses, mapes, mspes, corrs))

        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        valid_data, valid_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')
        path = os.path.join(self.args.checkpoints, setting)
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)
        writer = SummaryWriter('event/run_wind_speed/{}'.format(self.args.model_name))

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion(self.args.loss)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        if self.args.resume:
            self.model, lr, epoch_start = load_model(self.model, path, model_name=self.args.data, horizon=self.args.horizon)
        else:
            epoch_start = 0

        for epoch in range(epoch_start, self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_resnet1D(
                    train_data, batch_x, batch_y)
                true = true.mean(-1)
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    wandb.log({"Speed": speed})
                    wandb.log({"left_time ": left_time })
                    wandb.log({"loss": loss})
                
                if self.args.use_amp:
                    print('use amp')    
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            print('--------start to validate-----------')
            valid_loss = self.valid(valid_data, valid_loader, criterion)
            print('--------start to test-----------')
            test_loss = self.valid(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} valid Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, valid_loss, test_loss))

            writer.add_scalar('train_loss', train_loss, global_step=epoch)
            writer.add_scalar('valid_loss', valid_loss, global_step=epoch)
            writer.add_scalar('test_loss', test_loss, global_step=epoch)
            wandb.log({"Train Loss": train_loss})
            wandb.log({"Validation Loss": valid_loss})
            wandb.log({"Test Loss": test_loss})
            wandb.log({"Cost time": time.time()-epoch_time})
            
            early_stopping(valid_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            lr = adjust_learning_rate(model_optim, epoch+1, self.args)
            wandb.log({"Learning rate adjustent": lr})
            
        save_model(epoch, lr, self.model, path, model_name=self.args.data, horizon=self.args.pred_len)
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, evaluate=False):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        mids = []
        pred_scales = []
        true_scales = []
        mid_scales = []
        
        if evaluate:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_resnet1D(
                test_data, batch_x, batch_y)
            true = true.mean(-1)
            true_scale = true_scale.mean(-1)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            pred_scales.append(pred_scale.detach().cpu().numpy())
            true_scales.append(true_scale.detach().cpu().numpy())
            wandb.log({"Predicted Scales test": pred_scales})
            wandb.log({"True Scales test": true_scales})
            wandb.log({"Predicted test": pred})
            wandb.log({"True test": true})
        preds = np.array(preds)
        trues = np.array(trues)

        pred_scales = np.array(pred_scales)
        true_scales = np.array(true_scales)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        true_scales = true_scales.reshape(-1, true_scales.shape[-2], true_scales.shape[-1])
        pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])

        mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
        maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)
        
        # For true and Predicted
        wandb.log({"mae test": mae})
        wandb.log({"mse test": mse})
        wandb.log({"rmse test": rmse})
        wandb.log({"mape test": mape})
        wandb.log({"mspe test": mspe})
        wandb.log({"corr test": corr})
        
         # For true and Predicted scales
        wandb.log({"mae_scale test": maes})
        wandb.log({"mse_scale test": mses})
        wandb.log({"rmse_scale test": rmses})
        wandb.log({"mape_scale test": mapes})
        wandb.log({"mspe_scale test": mspes})
        wandb.log({"corr_scale test": corrs})
        print('normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
        print('TTTT Final --> denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mses, maes, rmses, mapes, mspes, corrs))

       

        # result save
        if self.args.save:
            folder_path = 'exp/wind_speed_forecast_results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
            print('Test:mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))

            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
            np.save(folder_path + 'pred_scales.npy', pred_scales)
            np.save(folder_path + 'true_scales.npy', true_scales)
            
        return mae, maes, mse, mses

    def _process_one_batch_resnet1D(self, dataset_object, batch_x, batch_y):
        batch_x = batch_x.double().to(self.device)
        batch_y = batch_y.double()
        outputs = self.model(batch_x)
        #if self.args.inverse:
        outputs_scaled = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        batch_y_scaled = dataset_object.inverse_transform(batch_y)
        return outputs, outputs_scaled, 0,0, batch_y, batch_y_scaled

