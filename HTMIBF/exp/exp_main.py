import json

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST,PatchTSM,PatchTSM2,PatchTSM4
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')







import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main2(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main2, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'PatchTSM':PatchTSM,
            'PatchTSM2':PatchTSM2,
            'PatchTSM4': PatchTSM4

        }
        self.beta = self.args.beta
        self.use_tqdm = not self.args.no_tqdm
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        num_params = sum(p.numel() for p in model.parameters())
        print(num_params)
        self.args.memory_track = False
        if self.args.memory_track:

            module_to_track = model.model.backbone  # 选择要监视的模块
            hook_handler = module_to_track.register_forward_hook(
                lambda module, input, output: print(
                    f"Memory usage of {module.__class__.__name__}: {torch.cuda.memory_allocated(self.device) / (1024 ** 2):.2f} MB"))
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):

        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,weight_decay=self.args.weight_decay)
        # model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate,weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        preds = []
        trues = []
        inputx = []
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder

                if self.args.model == "PatchTSM2":
                    outputs,KL_loss = self.model(batch_x)
                elif 'Linear' in self.args.model or 'TST' in self.args.model or 'TSM' in self.args.model:
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                # if self.args.model == "PatchTSM2":
                #     loss += KL_loss.cpu() * self.beta
                preds.append(pred.numpy())
                trues.append(true.numpy())
                inputx.append(batch_x.detach().cpu().numpy())
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))

        self.model.train()
        return total_loss, mse, mae

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path,exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)
        time_now = time.time()
        val_mae_list=[]
        test_mae_list=[]
        val_mse_list=[]
        test_mse_list=[]
        val_loss_list = []
        test_loss_list =[]

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            if self.use_tqdm:
                epoch_iter = tqdm(train_loader, desc="Iteration")
            else:
                epoch_iter = train_loader
            KL_loss = 0
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(epoch_iter):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder

                if self.args.model == "PatchTSM2":
                    outputs,KL_loss = self.model(batch_x)
                elif 'Linear' in self.args.model or 'TST' in self.args.model or 'TSM' in self.args.model:
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                # print(outputs.shape,batch_y.shape)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                if self.args.model == "PatchTSM2":
                    loss += KL_loss * self.beta
                train_loss.append(loss.item())

                if self.use_tqdm:
                    epoch_iter.set_description(f"train_loss: {loss.item():.4f},KL_loss:{KL_loss}")
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj in ['TST','TSM']:
                    adjust_learning_rate(model_optim, scheduler, epoch, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch, time.time() - epoch_time))




            train_loss = np.average(train_loss)
            vali_loss, vali_mse, vali_mae = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_mse, test_mae = self.vali(test_data, test_loader, criterion)
            val_mae_list.append(float(vali_mae))
            test_mae_list.append(float(test_mae))
            val_mse_list.append(float(vali_mse))
            test_mse_list.append(float(test_mse))
            val_loss_list.append(float(vali_loss))
            test_loss_list.append(float(test_loss))


            print(
                f"Epoch: {epoch} Steps: {train_steps} | Train Loss: {train_loss:.4f} Vali Loss: {vali_loss:.4f} Test Loss: {test_loss:.4f} |"
                f"vali_mse:{vali_mse},vali_mae:{vali_mae},test_mse:{test_mse},test_mae:{test_mae}")
            early_stopping(vali_loss, self.model, path, epoch)


            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj not in ['TST','TSM']:
                adjust_learning_rate(model_optim, scheduler, epoch, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'

        folder_path =  os.path.join(self.args.res_path,setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path,exist_ok=True)
        best_epoch = early_stopping.best_epoch
        res_record = json.dumps(
            {"val_mae": [val_mae_list[-1]],
             "val_mse": [val_mse_list[-1]],
             "test_last_mae": [test_mae_list[-1]],
             "test_last_mse": [test_mse_list[-1]],
             "test_best_mae": [test_mae_list[best_epoch]],
             "test_best_mse": [test_mse_list[best_epoch]],
             "val_mae_list": val_mae_list,
             "test__mae_list": test_mae_list,
             "val_mse_list": val_mse_list,
             "test__mse_list": test_mse_list,
             "val_loss_list":val_loss_list,
             "test_loss_list":test_loss_list})
        # 保存 JSON 数据到文件
        with open(os.path.join(folder_path,"result.json"), 'a') as json_file:
            json_file.write(res_record)
            json_file.write("\n")


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
        folder_path =  os.path.join(self.args.res_path,setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path,exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder

                if self.args.model == "PatchTSM2":
                    outputs,KL_loss = self.model(batch_x)
                elif 'Linear' in self.args.model or 'TST' in self.args.model or 'TSM' in self.args.model:
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])
        # result save
        folder_path =  os.path.join(self.args.res_path,setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path,exist_ok=True)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open(os.path.join(folder_path,"result.txt"), 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
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
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder


                if self.args.model == "PatchTSM2":
                    outputs,KL_loss = self.model(batch_x)
                elif 'Linear' in self.args.model or 'TST' in self.args.model or 'TSM' in self.args.model:
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
        folder_path =  os.path.join(self.args.res_path,setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path,exist_ok=True)

        np.save(folder_path + 'real_prediction.npy', preds)

        return


