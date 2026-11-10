import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        is_best = False
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            is_best = True
        return is_best

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path +'/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def prob_visual(data_mean, true, pred, down, up, random_int, name='./pic/test.pdf'):
    plt.figure(figsize=(15,4))
    plt.plot(pred, label='Prediction', linewidth=1.5, color='blue')
    # plt.plot(data_mean, label='DiffMean', linewidth=1.5, color='green')
    plt.plot(true, label='GroundTruth', linewidth=1.5, color='black')    
    plt.fill_between(np.arange(len(up))+96, down, up, color="green", alpha=0.2, label="Uncertainty Range")
    for i in range(len(random_int)):
        plt.axvline(x = random_int[i] + 96, color="grey", linestyle="--", linewidth=1)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()

def prob_visual(data, true, pred, random_int, name='./pic/test.pdf'):
    print('start ploting')
    plt.figure(figsize=(12,7))
    plt.grid(True) 
    pred = np.mean(data, axis=0)
    plt.plot(np.arange(192)+96, pred, label='Mean Prediction', linewidth=1.5, color='darkblue')
    # plt.plot(data_mean, label='DiffMean', linewidth=1.5, color='green')
    plt.plot(true, label='GroundTruth', linewidth=2., color='#9D2121')    
    # plt.fill_between(np.arange(len(up))+96, down, up, color="green", alpha=0.2, label="Uncertainty Range")
    percentiles = np.percentile(data, q=[2.5, 25, 75, 97.5], axis=0)
    plt.fill_between(np.arange(192)+96, percentiles[0], percentiles[3], color="#1f77b4", alpha=0.2, label="95% range")
    plt.fill_between(np.arange(192)+96, percentiles[1], percentiles[2], color="#0F4A74", alpha=0.2, label="50% Range")
    # for i in range(len(random_int)):
    #     plt.axvline(x = random_int[i] + 96, color="grey", linestyle="--", linewidth=1)
    # plt.ylim(-2.2, 0.5) # ETTh1
    # plt.ylim(-1.8, 0.) # ETTm1
    plt.ylim(-4, 7.0) #traffic
    # plt.ylim(-8.0, 7.5)
    
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()

# def pdf_visual(data, true, pred, name='./pic/test.pdf'):
#     std = np.std(data)
#     plt.figure()
#     sns.kdeplot(data, color="green", fill=True)
#     plt.axvline(x=true, color="black", linestyle="--", linewidth=2, label='Truth')
#     plt.axvline(x=pred, color="blue", linestyle="--", linewidth=2, label='pred')
#     plt.xlabel(f"Value  std:{std}")
#     plt.ylabel("Density")
#     plt.legend()
#     plt.savefig(name, bbox_inches='tight')

def pdf_visual(data, true, pred, random_int, name='./pic/test.pdf'):
    plt.figure(figsize=(16,16))
    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)
    _, _, s = DDN(true, 15)
    # random_int = np.random.randint(0,len(true), size=4)
    for i in range(9):
        plt.subplot(3,3,i+1)
        std = np.std(data[:, random_int[i]])        
        sns.kdeplot(data[:, random_int[i]], color="green", fill=True)
        plt.axvline(x=true[random_int[i]], color="black", linestyle="--", linewidth=2, label='Truth')
        plt.axvline(x=pred[random_int[i]], color="blue", linestyle="--", linewidth=2, label='pred')
        plt.xlabel(f"Value{i+1}  Std:{std:.3f} Max-Min:{data_max[random_int[i]]-data_min[random_int[i]]:.3f} S:{s[random_int[i]]:.3f}")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(name, bbox_inches='tight')
    plt.close()

def uncertain_visual(data, true, pred, name='./pic/test.pdf'):
    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)
    intervals = data_max - data_min
    data_std = np.std(data, axis=0)
    _, _, s = DDN(true, 33)    
    # uncertain = np.abs(true - pred)
    uncertain = (np.mean(data, axis=0) - true)
    s = np.std(uncertain, axis=0)
    plt.figure(figsize=(12, 6))
    plt.plot(intervals, label='max-min', linewidth=1.5, color='blue')
    plt.plot(data_std, label='diffusion_std', linewidth=1.5, color='red')
    # plt.plot(s, label='std of y', linewidth=1.5, color='green')
    plt.axhline(y=s, color='green', linestyle='--', linewidth=1.5, label="std of res")
    plt.plot(uncertain, label='y - y^', linewidth=1.5, color='black')
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()

def DDN(data, kernel):
    x = torch.tensor(data)
    x_window = x.unfold(-1, kernel, 1)
    m, s = x_window.mean(dim=-1).numpy(), x_window.std(dim=-1).numpy()
    m, s = np.pad(m, (kernel//2,kernel//2), mode='edge'), np.pad(s, (kernel//2,kernel//2), mode='edge')
    data = (data - m) / (s + 1e-5)
    return data, m, s

def acf_visual(data, true, pred, name='./pic/test.pdf'):
    uncertain = np.abs(np.mean(data, axis=0) - true)
    plt.figure(figsize=(10, 4))
    plot_acf(uncertain, lags=30)
    plt.savefig(name, bbox_inches="tight")
    plt.close()

def scenarios_visual(data, true, name='./pic/test.pdf'):
    plt.figure(figsize=(12,7))
    # pred = np.mean(data, axis=0)
    # plt.plot(np.arange(192)+96, pred, label='Mean Prediction', linewidth=1.5, color='olivedrab')
    # plt.plot(data_mean, label='DiffMean', linewidth=1.5, color='green')
    plt.plot(true, label='GroundTruth', linewidth=1.5, color='black')   
    # print(data[:,0]) 
    for i in range(5):
        plt.plot(np.arange(192)+96, data[i], linewidth=1.5, alpha=0.3, color='blue')
    # plt.ylim(-2.5, 1.0) # ETTh1
    # plt.ylim(-2.0, 0.5) # ETTm1
    # plt.ylim(-7.5, 10.0) #traffic
    # plt.ylim(-8.0, 7.5)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()