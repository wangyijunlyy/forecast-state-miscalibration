from data_provider.data_factory import data_provider

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from utils.tools import prob_visual, pdf_visual, uncertain_visual, acf_visual, scenarios_visual
import gc
# from model9_NS_transformer.ns_models import ns_Transformer
from model_ppm.exp.exp_basic import Exp_Basic
from model_ppm.PPMF import MLP_PPM
import torch.distributed as dist
import numpy as np
from math import sqrt
import torch
import torch.nn as nn
from torch import optim
import os
import time
from utils.metrics import  calc_quantile_CRPS_sum
from multiprocessing import Pool
import CRPS.CRPS as pscore
import warnings
import math

warnings.filterwarnings('ignore')


def ccc(id, pred, true):

    res_box = np.zeros(len(true))

    for i in range(len(true)):
        res = pscore(pred[i], true[i]).compute()
        res_box[i] = res[0]
    return res_box


def calculate_crps_sum_worker(args):
        pred, true = args
        p_in = np.sum(pred, axis=-1).T
        t_in = np.sum(true, axis=-1).reshape(-1)
        crps = ccc(8, p_in, t_in)
        return crps.mean()
        
def calculate_crps_worker(pred, true):
        p_in = pred.transpose(1, 0, 2)
        t_in = true
        all_res = []
        for i in range(pred.shape[-1]):
            crps = ccc(8, p_in[:,:,i], t_in[:,i])
            all_res.append(crps)
        all_res= np.array(all_res)
        if isinstance(all_res, np.ndarray):
            return np.mean(all_res, axis=0).mean()
        else:
            return all_res


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)        
        

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, mode='Model'):
        if mode == 'Model':
            model_optim = optim.Adam([
                {'params': self.model.parameters(), 'lr': self.args.learning_rate},      # 主模型
            ])
        else:
            model_optim = None
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion  

    
    def kde_loss(self, pred, true, tau=None, kernel="gaussian", nu=3.0, eps=1e-12):
        """
        pred: [B, K, L, N]
        true: [B, L, N]
        kernel: one of {"gaussian","laplace","student_t","cauchy","logistic","epanechnikov"}
        nu: df for student_t kernel
        """
        h = float(self.args.var)  # bandwidth
        B, K, L, N = pred.shape

        # u = (y - yk) / h
        diff = pred - true.unsqueeze(1)            # [B,K,L,N]
        u = diff / h

        # --- compute log K(u) ---
        if kernel == "gaussian":
            # K(u)= (1/sqrt(2pi)) exp(-u^2/2)
            log_kernel = -0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)

        elif kernel == "laplace":    # var=0.2
            # K(u)= 0.5 exp(-|u|)
            log_kernel = -u.abs() - math.log(2.0)

        elif kernel == "student_t":
            # K(u)= t_pdf(u; nu) = C * (1 + u^2/nu)^(-(nu+1)/2)
            # logC = log Gamma((nu+1)/2) - log Gamma(nu/2) - 0.5*log(nu*pi)
            nu_t = torch.as_tensor(nu, device=pred.device, dtype=pred.dtype)
            logC = (torch.lgamma((nu_t + 1.0) / 2.0)
                    - torch.lgamma(nu_t / 2.0)
                    - 0.5 * (torch.log(nu_t) + math.log(math.pi)))
            log_kernel = logC - 0.5 * (nu_t + 1.0) * torch.log1p(u.pow(2) / nu_t)

        elif kernel == "cauchy":
            # K(u)= 1 / (pi * (1 + u^2))
            log_kernel = -math.log(math.pi) - torch.log1p(u.pow(2))

        elif kernel == "logistic":
            # K(u)= exp(-u) / (1 + exp(-u))^2
            # stable log: -u - 2*softplus(-u)
            log_kernel = -u - 2.0 * torch.nn.functional.softplus(-u)

        elif kernel == "epanechnikov":
            # K(u)= 0.75*(1-u^2) for |u|<=1 else 0
            inside = (u.abs() <= 1.0)
            base = 0.75 * (1.0 - u.pow(2))
            base = torch.clamp(base, min=eps)  # avoid log(0) inside region
            log_kernel = torch.full_like(u, float("-inf"))
            log_kernel[inside] = torch.log(base[inside])
        
        elif kernel in ["uniform", "boxcar", "rect"]:
            # Uniform/Boxcar kernel:
            # K(u) = 0.5 * I(|u| <= 1)   (normalized on [-1, 1])
            inside = (u.abs() <= 1.0)
            log_kernel = torch.full_like(u, float("-inf"))
            log_kernel[inside] = -math.log(2.0)

        elif kernel == "cosine":
            # Cosine kernel:
            # K(u) = (pi/4) * cos(pi*u/2) * I(|u| <= 1)
            # normalized on [-1, 1]
            inside = (u.abs() <= 1.0)
            log_kernel = torch.full_like(u, float("-inf"))

            # cos term is >= 0 on [-1,1], but hits 0 at |u|=1, so clamp for log stability
            cos_term = torch.cos((math.pi / 2.0) * u)
            cos_term = torch.clamp(cos_term, min=eps)

            log_kernel[inside] = math.log(math.pi / 4.0) + torch.log(cos_term[inside])


        else:
            raise ValueError(f"Unknown kernel: {kernel}")


        log_sum = torch.logsumexp(log_kernel, dim=1)        # [B,L,N]
        log_p_elem = log_sum - (math.log(K) + math.log(h))  # [B,L,N]

        log_p = log_p_elem.mean(dim=(-1, -2))               # [B]
        log_p = torch.clamp(log_p, min=-25)                 # 防 underflow（可按需调）
        loss = -log_p.mean()
        return loss

    
    
    def mean_loss(self, pred, true):
        # pred = pred.squeeze(1)
        mean_pred = torch.mean(pred, dim=1)
        loss = (mean_pred - true)**2

        return loss.mean()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder

                if self.args.model in ['ns_Transformer']:
                    output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.args.K)
                else:
                    output = self.model(batch_x, self.args.K)

                batch_y=batch_y[:, -self.args.pred_len:, :]                

                f_dim = -1 if self.args.features == 'MS' else 0
                output = output[:, :, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = self.args.tau * self.kde_loss(output, batch_y, self.args.tau, kernel='gaussian', nu=self.args.nu) + self.mean_loss(output, batch_y)

                loss = loss.detach().cpu()
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        condition_path=os.path.join(self.args.pretrain_checkpoints, setting)

        if not os.path.exists(path): 
            os.makedirs(path)
        # if not os.path.exists(condition_path):
        #     os.makedirs(condition_path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()

        criterion = self._select_criterion()
           
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):

            epoch_time = time.time()

            iter_count = 0
            train_loss = []
            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
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

                n = batch_x.size(0)
                
                if self.args.model in ['ns_Transformer']:
                    output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.args.K)
                else:
                    output = self.model(batch_x, self.args.K)

                batch_y=batch_y[:, -self.args.pred_len:, :]                

                f_dim = -1 if self.args.features == 'MS' else 0
                output = output[:, :, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = self.args.tau * self.kde_loss(output, batch_y, self.args.tau, kernel='gaussian', nu=self.args.nu) + self.mean_loss(output, batch_y)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
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

                a = 0


            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss= self.vali(vali_data, vali_loader, criterion)
            test_loss= self.vali(test_data, test_loader, criterion)


            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}  Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping( vali_loss, self.model, path)

            if (math.isnan(train_loss)):
                break

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        
        return self.model


    def test(self, setting, test=1):
        #####################################################################################################
        ########################## local functions within the class function scope ##########################

        def compute_true_coverage_by_gen_QI(n_bin, dataset_object, all_true_y, all_generated_y):
            n_bins = n_bin
            quantile_list = np.arange(n_bins + 1) * (100 / n_bins)
            # compute generated y quantiles
            y_pred_quantiles = np.percentile(all_generated_y.squeeze(), q=quantile_list, axis=1)
            y_true = all_true_y.T
            quantile_membership_array = ((y_true - y_pred_quantiles) > 0).astype(int)
            y_true_quantile_membership = quantile_membership_array.sum(axis=0)
            # y_true_quantile_bin_count = np.bincount(y_true_quantile_membership)
            y_true_quantile_bin_count = np.array(
                [(y_true_quantile_membership == v).sum() for v in np.arange(n_bins + 2)])

            # combine true y falls outside of 0-100 gen y quantile to the first and last interval
            # y_true_quantile_bin_count[1] += y_true_quantile_bin_count[0] #cheat
            # y_true_quantile_bin_count[-2] += y_true_quantile_bin_count[-1]
            y_true_quantile_bin_count_ = y_true_quantile_bin_count[1:-1]
            # compute true y coverage ratio for each gen y quantile interval
            # y_true_ratio_by_bin = y_true_quantile_bin_count_ / dataset_object.test_n_samples
            y_true_ratio_by_bin = y_true_quantile_bin_count_ / dataset_object
            # assert np.abs(
            #     np.sum(y_true_ratio_by_bin) - 1) < 1e-10, "Sum of quantile coverage ratios shall be 1!"
            qice_coverage_ratio = np.absolute(np.ones(n_bins) / n_bins - y_true_ratio_by_bin).mean()
            return y_true_quantile_bin_count_, qice_coverage_ratio, y_true
        

        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        minibatch_sample_start = time.time()

        self.model.eval()
        total_mse=0.0
        total_mae=0.0
        total_samples=0.0
        sum_crps = 0.0
        sum_crps_sum =0.0 
        # QICE
        bin_count = np.zeros(10)
        QICE_samples = 0
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

                if self.args.model in ['ns_Transformer']:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, 100)
                else:
                    outputs = self.model(batch_x, 100)                

                outputs = outputs.detach().cpu().numpy()[:, :, -self.args.pred_len:, :]


                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()


                ### QICE
                B,S,L,N = pred.shape
                q_pred = pred.transpose(0,2,3,1).reshape(-1, S)
                q_true = true.reshape(-1,1)

                batch_bin_count, _, _ = compute_true_coverage_by_gen_QI(
                    n_bin=10, dataset_object=1,
                    all_true_y=q_true, all_generated_y=q_pred, )
                
                bin_count += batch_bin_count
                QICE_samples += q_pred.shape[0]

                ### crps
                batch_crps = self.calculate_batch_crps(pred, true)
                # batch_crps = 0
                sum_crps += batch_crps

 
                pred_ns =np.mean(pred,axis=1)
                # print('test2 shape:', pred_ns.shape, true.shape)
                mae, mse, rmse, mape, mspe = metric(pred_ns, true)
                # print('mae_mse',mae, mse)
                total_mse += mse * pred_ns.shape[0]
                total_mae += mae * pred_ns.shape[0]
                total_samples += pred_ns.shape[0]

                # preds.append(pred.sum(-1))                     
                # trues.append(true.sum(-1))
                preds.append(pred)                     
                trues.append(true)

                voutput = outputs
                del outputs 
                gc.collect() 

                if i % 10 == 0 and i != 0:
                    print()
                    print('Testing: %d/%d cost time: %f min' % (
                        i, len(test_loader), (time.time() - minibatch_sample_start) / 60))
                    minibatch_sample_start = time.time()


        print('total_samples',total_samples)
        avg_crps = sum_crps / total_samples
        mse_total = total_mse / total_samples
        mae_total = total_mae / total_samples
        print('NT metrc: CRPS:{:.4f}'.format(avg_crps))
        print('NT metrc: mse:{:.4f}, mae:{:.4f} '.format(mse_total, mae_total))
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        # QICE
        n_bin = 10
        qice_coverage_ratio = bin_count / QICE_samples
        print('QICE bin count:', qice_coverage_ratio)
        qice_coverage_ratio = np.absolute(np.ones(n_bin)/n_bin - qice_coverage_ratio).mean()

        print('CARD metrc: QICE:{:.4f}%'.format(qice_coverage_ratio * 100))

        # f = open("result.txt", 'a')
        f = open("./results/" + self.args.data_name + ".txt", 'a')
        # print('writting')
        f.write(setting + "  \n")
        f.write('tau:{}, var:{}, K:{}, QICE:{:.4f}%, CRPS:{:.4f}, mse:{:.4f}, mae:{:.4f}'.format(self.args.tau, self.args.var, self.args.K, qice_coverage_ratio * 100, avg_crps, mse_total, mae_total))
        f.write('\n')
        f.write('\n')
        f.close()

    def calculate_batch_crps(self, pred, true):
        B,S,L,N = pred.shape
        all_res = []
        for i in range(B):
            p_in = pred[i]
            t_in = true[i]
            p_in = p_in.transpose(1,2,0).reshape(-1, S)
            t_in = t_in.reshape(-1)
            crps = ccc(i, p_in, t_in)
            all_res.append(crps)
        # for i in range(N):
        #     p_in = pred[...,i]
        #     t_in = true[...,i]
        #     p_in = p_in.transpose(0,2,1).reshape(-1, S)
        #     t_in = t_in.reshape(-1)
        #     crps = ccc(i, p_in, t_in)
        #     print(crps.mean())
        #     all_res.append(crps)
        # assert 0
        all_res= np.array(all_res)
        if isinstance(all_res, np.ndarray):
            return np.mean(all_res, axis=1).sum()
        else:
            return all_res
        