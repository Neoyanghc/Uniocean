from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model_Informer import InformerUni, Informer,  Informer_two
from models.model_Autoformer import AutoformerUni,Autoformer
from models.model_FEDformer import FEDformerUni,FEDformer
from models.model_base import ConvLSTM,GRU

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')

class Exp_UniOcean(Exp_Basic):
    def __init__(self, args):
        super(Exp_UniOcean, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informerUniOcean':InformerUni,
            'informer':Informer,
            'informer_two':Informer_two,
            'autoformerUniOcean':AutoformerUni,
            'autoformer':Autoformer,
            'fedformerUniOcean':FEDformerUni,
            'fedformer':FEDformer,
            'convlstm':ConvLSTM,
            'gru':GRU,
        }
        if self.args.model =='convlstm':
            e_layers = self.args.e_layers
            model = model_dict[self.args.model](
                self.args.input_dim,
                self.args.hidden_dim,
                self.args.kernel_size, 
                self.args.num_layers,
                self.args.seq_len,
                self.args.batch_first, 
                self.args.bias,
                self.args.return_all_layers,
                self.device
            ).float()
        if self.args.model =='gru':
             e_layers = self.args.e_layers
             model = model_dict[self.args.model](
                 self.args.enc_in,
                 self.args.d_model, 
                 self.args.num_layers,
             )
        if self.args.model=='informer' or self.args.model=='informerUniOcean' or self.args.model=='informer_two' or self.args.model=='autoformerUniOcean' or self.args.model=='fedformerUniOcean' or self.args.model=='autoformer' or self.args.model=='fedformer':
            e_layers = self.args.e_layers if self.args.model=='informer' or self.args.model=='informerUniOcean'or self.args.model=='informer_two' or  self.args.model=='autoformerUniOcean'or self.args.model=='fedformerUniOcean' or self.args.model=='autoformer' or self.args.model=='fedformer'  else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.move_avg,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.args.use_multi_scale,
                self.args.patembed,
                self.args.scales,
                self.args.scale_factor,
                self.args.version,
                self.args.mode_select,
                self.args.modes,
                self.args.L,
                self.args.base,
                self.args.cross_activation,
                self.args.conv_dff,
                self.device
                
            ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            model = model.cuda()
            total = sum([param.nelement() for param in model.parameters()])
            print('total param:' ,total)

        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'custom':Dataset_Custom,
            'SST5': Dataset_Custom,
            'SST6': Dataset_Custom,
            'SST7': Dataset_Custom,
            'SAL5': Dataset_Custom,
            'SAL6': Dataset_Custom,
            'SAL8': Dataset_Custom,
            'OHC1': Dataset_Custom,
            'OHC5': Dataset_Custom,
            'ICEC1': Dataset_Custom,
            'OC1': Dataset_Custom,
            'OISSS1': Dataset_Custom,
            'OISSS3': Dataset_Custom,
            'OISST1': Dataset_Custom,
            'OISST2': Dataset_Custom,
            'OISST3': Dataset_Custom,
            'OISST4': Dataset_Custom,
            'OISST5': Dataset_Custom,
            'ALL1': Dataset_Custom,
            'ALL2': Dataset_Custom,
            'ALL3': Dataset_Custom,
            'ALL4': Dataset_Custom,
            'ALL5': Dataset_Custom,
            'ALL6': Dataset_Custom,
        }
        Data = data_dict[self.args.data]
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
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            # drop_last=drop_last,
            drop_last=True)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss1 = []
        total_loss2 = []
        total_loss3 = []
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,_) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            
            if self.args.root_path=='./data/ALL/':
                if self.args.data=='ALL2':
                    loss1 = criterion(pred[:,:,:,0].detach().cpu(), true[:,:,:,0].detach().cpu())
                    loss2 = criterion(pred[:,:,:,1].detach().cpu(), true[:,:,:,1].detach().cpu())
                    loss3 = criterion(pred[:,:,:,2].detach().cpu(), true[:,:,:,2].detach().cpu())
                    total_loss1.append(loss1)
                    total_loss2.append(loss2)
                    total_loss3.append(loss3)
                else:
                    loss1 = criterion(pred[:,:,:,0].detach().cpu(), true[:,:,:,0].detach().cpu())
                    loss2 = criterion(pred[:,:,:,1].detach().cpu(), true[:,:,:,1].detach().cpu())
                    total_loss1.append(loss1)
                    total_loss2.append(loss2)
            else:
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss)

        if self.args.root_path=='./data/ALL/':
            if self.args.data=='ALL2':
                total_loss1 = np.average(total_loss1)
                total_loss2 = np.average(total_loss2)
                total_loss3 = np.average(total_loss3)
                total_loss=(total_loss1+total_loss2+total_loss3)/3
            else:
                total_loss1 = np.average(total_loss1)
                total_loss2 = np.average(total_loss2)
                total_loss=(total_loss1+total_loss2)/3
        else:
            total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)

        ############ 
        if self.args.multi_task:
            if not os.path.exists(path):
                os.makedirs(path)
            model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(model_path))

        if not os.path.exists(path):
            os.makedirs(path)
        

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_loss1 = []
            train_loss2 = []
            train_loss3 = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,_) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()

                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)

                if self.args.root_path=='./data/ALL/':
                    if self.args.data=='ALL2':
                        loss1=criterion(pred[:,:,:,0], true[:,:,:,0])
                        loss2=criterion(pred[:,:,:,1], true[:,:,:,1])
                        loss3=criterion(pred[:,:,:,2], true[:,:,:,2])
                        train_loss1.append(loss1.item())
                        train_loss2.append(loss2.item())
                        train_loss3.append(loss3.item())
                        # loss=loss1+loss2+loss3
                        loss=(loss1+loss2+loss3)/3
                    else:
                        loss1=criterion(pred[:,:,:,0], true[:,:,:,0])
                        loss2=criterion(pred[:,:,:,1], true[:,:,:,1])
                        train_loss1.append(loss1.item())
                        train_loss2.append(loss2.item())
                        loss=(loss1+loss2)/2
                    
                    train_loss.append(loss.item())
                else:
                    loss = criterion(pred, true)
                    train_loss.append(loss.item()) 
                
                if (i+1) % 100==0:
                    # if self.args.data=='ALL1':
                    #     print("\titers: {0}, epoch: {1} | lossSAL: {2:.7f} | lossSST: {3:.7f} | lossOHC: {4:.7f}".format(i + 1, epoch + 1, loss1.item(),loss2.item(),loss3.item()))
                    # else:
                    #     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            if self.args.root_path=='./data/ALL/':

                train_loss = np.average(train_loss)
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)

            else: 
                train_loss = np.average(train_loss)
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,_) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
        
        preds = np.array(preds)
        trues = np.array(trues)
        
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # mae, mse, rmse, mape, mspe = metric(preds, trues)
        if self.args.root_path=='./data/ALL/':
            if self.args.data=='ALL2':
                mae1, mse1,_,_,_ = metric(preds[:,:,0], trues[:,:,0])
                mae2, mse2,_,_,_ = metric(preds[:,:,1], trues[:,:,1])
                mae3, mse3,_,_,_ = metric(preds[:,:,2], trues[:,:,2])
            else:
                mae1, mse1,_,_,_ = metric(preds[:,:,0], trues[:,:,0])
                mae2, mse2,_,_,_ = metric(preds[:,:,1], trues[:,:,1])
            
            if self.args.data=='ALL2':
                print('SSS mse:{}, mae:{}'.format(mse1,mae1))
                print('SST mse:{}, mae:{}'.format(mse2,mae2))
                print('OHC mse:{}, mae:{}'.format(mse3,mae3))
                np.save(folder_path+'metrics.npy', np.array([mae1, mse1,mse2,mae2,mse3,mae3]))
            else:
                print('data1 mse:{}, mae:{}'.format(mse1,mae1))
                print('data2 mse:{}, mae:{}'.format(mse2,mae2))
                np.save(folder_path+'metrics.npy', np.array([mae1, mse1,mse2,mae2]))
        else:
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('mse:{}, mae:{}'.format(mse, mae))
            np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))

        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        return
    
    def get_prediction(self, setting):
        
        test_data, test_loader = self._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        self.model.eval()
        
        preds = []
        trues = []
        times = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_y_time) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            times.append(batch_y_time.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        times = np.array(times)
        
        print('shape:', preds.shape, trues.shape, times.shape)
    
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'pred_.npy', preds)
        np.save(folder_path+'true_.npy', trues)
        np.save(folder_path+'time_.npy', times)
        
        return 

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,_) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding==0:
            if self.args.root_path=='./data/ALL/':
                if self.args.data=='ALL2':
                    dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-2],3]).float()
                else:
                    dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-2],2]).float()
            else:
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            if self.args.root_path=='./data/ALL/':
                if self.args.data=='ALL2':
                    dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-2],3]).float()
                else:
                    dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-2],2]).float()
            else:
                dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder

        ## @cyw 2023.12
        ### 联合训练
        if self.args.root_path=='./data/ALL/':
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if self.args.inverse:
                outputs = dataset_object.inverse_transform(outputs)
            if self.args.get_prediction:
                outputs = dataset_object.inverse_transform(outputs)
                batch_y = dataset_object.inverse_transform(batch_y)
                
            f_dim = -1 if self.args.features=='MS' else 0
            batch_y = batch_y[:,-self.args.pred_len:,f_dim:,:].to(self.device)
                
        ###独自训练
        else:
            if self.args.model=='convlstm' or self.args.model=='gru' :
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():outputs = self.model(batch_x)[0]
                else:
                    outputs = self.model(batch_x)[0]
                    
                if self.args.get_prediction:
                    outputs = dataset_object.inverse_transform(outputs)
                    batch_y = dataset_object.inverse_transform(batch_y)
            else:
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs,_ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs,_ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if self.args.inverse:
                    outputs = dataset_object.inverse_transform(outputs)
                
                if self.args.get_prediction:
                    outputs = dataset_object.inverse_transform(outputs)
                    batch_y = dataset_object.inverse_transform(batch_y)
                
                    

            f_dim = -1 if self.args.features=='MS' else 0
            batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        return outputs, batch_y
