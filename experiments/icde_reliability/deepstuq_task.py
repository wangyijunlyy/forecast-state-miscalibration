"""Full author DeepSTUQ architecture, pretraining, AWA, temperature and paired tests."""
import argparse
import configparser
import json
import os
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.special import ndtri

ROOT = Path(os.environ.get('ICDE_RUN_ROOT', '/home/wyj/icde_reliability_20261102'))
CODE = Path(__file__).resolve().parent
REPO = ROOT/'third_party/DeepSTUQ_Pytorch'
sys.path.insert(0, str(REPO))
from model.AGCRN_UQ import AGCRN_UQ
from model.BasicTrainer import Trainer
import model.train_methods as native_training
import lib.dataloader as native_data
from lib.TrainInits import init_seed
from lib.utils import enable_dropout
sys.path.insert(0, str(CODE))
from deepstuq_math import marginal_parameters, native_gaussian_nll
from reliability_metrics import ReliabilityMetrics, TargetReservoir, state_ids
from run_task import fit_dc, fit_models, CalibrationReservoir, atomic

VERSION = 'full-deepstuq-icde2023-v1'


def configuration(dataset, seed):
    config = configparser.ConfigParser(); config.read(REPO/'model'/f'{dataset}_AGCRN.conf')
    types = dict(num_nodes=int, lag=int, horizon=int, val_ratio=float, test_ratio=float,
        tod=lambda x:x == 'True', column_wise=lambda x:x == 'True', default_graph=lambda x:x == 'True',
        input_dim=int, output_dim=int, embed_dim=int, rnn_units=int, num_layers=int, cheb_order=int,
        p1=float, batch_size=int, epochs=int, lr_init=float, lr_decay=lambda x:x == 'True',
        lr_decay_rate=float, early_stop=lambda x:x == 'True', early_stop_patience=int,
        grad_norm=lambda x:x == 'True', max_grad_norm=int, real_value=lambda x:x == 'True', log_step=int)
    values = {key:types.get(key, str)(value) for section in config.sections() for key, value in config[section].items()}
    values.update(dataset=dataset, seed=seed, device='cuda:0', cuda=True, model_name='combined',
                  debug=True, teacher_forcing=False, cheb_k=values['cheb_order'])
    return SimpleNamespace(**values)


def loaders(args):
    original = native_data.normalize_dataset
    def historical_normalize(data, normalizer, column_wise=False):
        cut = len(data)-int(len(data)*(args.val_ratio+args.test_ratio))
        _, scaler = original(data[:cut], normalizer, column_wise)
        return scaler.transform(data), scaler
    native_data.normalize_dataset = historical_normalize
    try:
        return native_data.get_dataloader(args, normalizer=args.normalizer, tod=args.tod, dow=False,
                                          weather=False, single=False)
    finally:
        native_data.normalize_dataset = original


def train(args, out, data):
    model = AGCRN_UQ(args).cuda()
    for parameter in model.parameters():
        torch.nn.init.xavier_uniform_(parameter) if parameter.dim() > 1 else torch.nn.init.uniform_(parameter)
    train_loader, val_loader, test_loader, scaler = data
    if (out/'training.json').exists():
        record = json.loads((out/'training.json').read_text())
        if not record['training_complete'] or record['awa_epochs'] != 20:
            raise ValueError('Incomplete baseline checkpoint')
        model.load_state_dict(torch.load(out/'model.pt', map_location='cuda:0'))
        return model, record
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init, eps=1e-8, weight_decay=1e-6, amsgrad=False)
    trainer = Trainer(model, torch.nn.L1Loss().cuda(), optimizer, train_loader, val_loader, test_loader, scaler, args)
    history = []; original_train = trainer.train_epoch; original_val = trainer.val_epoch
    def train_epoch(epoch):
        value = original_train(epoch)
        if not np.isfinite(value):
            raise ValueError('Nonfinite pretraining objective')
        history.append(dict(epoch=epoch, train_loss=value)); return value
    def val_epoch(epoch, loader):
        value = original_val(epoch, loader)
        if not np.isfinite(value):
            raise ValueError('Nonfinite checkpoint-selection objective')
        history[-1]['validation_loss'] = value
        atomic(out/'training_history.json', history); return value
    trainer.train_epoch = train_epoch; trainer.val_epoch = val_epoch
    start = time.monotonic(); trainer.train()
    torch.save(trainer.model.state_dict(), out/'pretrained.pt')
    pretraining_seconds = time.monotonic()-start
    # Bind the undefined global referenced by the author function; preserve its full optimizer/schedule.
    native_training.model = trainer.model
    start = time.monotonic()
    averaged = native_training.awa_train_combined(trainer, epoch_swa=20)
    if int(averaged.n_averaged) != 10:
        raise ValueError('AWA did not produce all ten snapshots')
    model = averaged.module; torch.save(model.state_dict(), out/'model.pt')
    record = dict(version=VERSION, training_complete=True, args=vars(args),
        pretraining_epochs=len(history), best_pretraining_epoch=min(history, key=lambda r:r['validation_loss'])['epoch'],
        pretraining_seconds=pretraining_seconds, awa_epochs=20, awa_snapshots=int(averaged.n_averaged),
        awa_seconds=time.monotonic()-start, architecture_source=str(REPO/'model/AGCRN_UQ.py'),
        author_revision='4f2281531d07b48ba81aa1c519a90e65a71993e2', normalization='training segment only')
    atomic(out/'training.json', record)
    return model, record


def predict(model, batch, args, temperature):
    x, target = batch; model.eval(); enable_dropout(model)
    means = []; log_vars = []
    with torch.no_grad():
        for _ in range(10):
            mu, log_var = model(x[..., :args.input_dim], target, teacher_forcing_ratio=0.)
            means.append(mu.squeeze(-1).cpu().numpy()); log_vars.append(log_var.squeeze(-1).cpu().numpy())
    means = np.stack(means); log_vars = np.stack(log_vars)
    mu, variance = marginal_parameters(means, log_vars, temperature)
    _, raw_variance = marginal_parameters(means, log_vars, 0.)
    grid = ndtri((np.arange(100)+.5)/100)[None, :, None, None]
    return (mu[:, None]+np.sqrt(variance)[:, None]*grid,
            mu[:, None]+np.sqrt(raw_variance)[:, None]*grid, mu, variance)


def wait_for_primary():
    """Baseline training may overlap drained GPUs; inference stays subsequent."""
    announced = False
    while True:
        status = json.loads((ROOT/'status.json').read_text())
        pids = [status['controller_pid']]+[v['pid'] for v in status['active'].values()]
        live = False
        for pid in pids:
            try:
                command = (Path('/proc')/str(pid)/'cmdline').read_bytes()
                live |= str(ROOT/'code/launch.py').encode() in command or str(ROOT/'code/run_task.py').encode() in command
            except FileNotFoundError:
                pass
        if status['complete'] == 237 and not live:
            return
        if not announced:
            print('TRAINING_FINISHED_WAITING_FOR_PRIMARY_INFERENCE', flush=True); announced = True
        time.sleep(30)


def run(dataset, seed):
    out = ROOT/'deepstuq/runs'/f'{dataset}_seed{seed}'; out.mkdir(parents=True, exist_ok=True)
    if (out/'results.json').exists():
        return
    args = configuration(dataset, seed); init_seed(seed); os.chdir(ROOT/'deepstuq')
    data = loaders(args); model, training = train(args, out, data)
    wait_for_primary()
    _, val_loader, test_loader, scaler = data
    if (out/'temperature.json').exists():
        temperature = json.loads((out/'temperature.json').read_text())['log_temperature']
    else:
        init_seed(seed+10000)
        start = time.monotonic()
        temperature = float(native_training.train_cali_mc(model, 10, args, val_loader, scaler).detach().cpu().item())
        if not np.isfinite(temperature):
            raise ValueError('Nonfinite native calibration parameter')
        atomic(out/'temperature.json', dict(log_temperature=temperature, seconds=time.monotonic()-start,
            mc_samples=10, lbfgs_step=.02, lbfgs_max_iter=500, source='unmodified author train_cali_mc'))
    pool = CalibrationReservoir(50000, seed+20000); target_pool = TargetReservoir(4096, seed+21000)
    init_seed(seed+20000)
    for step, batch in enumerate(val_loader, 1):
        x, _, _, _ = predict(model, batch, args, temperature)
        y = scaler.inverse_transform(batch[1][..., :args.output_dim]).squeeze(-1).cpu().numpy()
        target_pool.update(y)
        if step <= 100:
            pool.update(x, y)
        if step % 20 == 0:
            print('CALIBRATION', dataset, seed, step, flush=True)
    rows, y, h, c = pool.arrays()
    task = dict(horizon=args.horizon)
    models, fit_meta, fit_seconds = fit_models(task, rows, y, h, c)
    models = {name: models[name] for name in ['DC16', 'GlobalDC', 'GlobalPIT']}
    with (out/'calibrators.pkl').open('wb') as handle:
        pickle.dump(models, handle)
    thresholds = target_pool.thresholds()
    names = ['DeepSTUQ', 'DeepSTUQ+DC16', 'DeepSTUQ+GlobalDC', 'DeepSTUQ+PIT', 'NoTemperature']
    accumulators = {name:ReliabilityMetrics(thresholds) for name in names}
    native = dict(n=0, nll_sum=0., interval_covered=0, interval_width_sum=0.)
    init_seed(seed+30000); windows = steps = 0; start = time.monotonic()
    for step, batch in enumerate(test_loader, 1):
        x, raw, mu, variance = predict(model, batch, args, temperature)
        y = scaler.inverse_transform(batch[1][..., :args.output_dim]).squeeze(-1).cpu().numpy()
        groups = state_ids(x, models['DC16'].mean_edges, models['DC16'].scale_edges)
        for name in names:
            if name == 'DeepSTUQ': values = x
            elif name == 'NoTemperature': values = raw
            else: values = models[{'DeepSTUQ+DC16':'DC16', 'DeepSTUQ+GlobalDC':'GlobalDC', 'DeepSTUQ+PIT':'GlobalPIT'}[name]].transform(x)
            accumulators[name].update(values, y, groups, ordered=True)
        width = 1.96*np.sqrt(variance)
        native['n'] += y.size
        native['nll_sum'] += float(native_gaussian_nll(mu, y, variance).sum())
        native['interval_covered'] += int((np.abs(y-mu) <= width).sum())
        native['interval_width_sum'] += float((2*width).sum())
        windows += len(y); steps = step
        if step % 20 == 0:
            print('TEST', dataset, seed, step, windows, flush=True)
    if steps != len(test_loader) or windows != len(test_loader.dataset):
        raise ValueError('Incomplete native DeepSTUQ test iterator')
    atomic(out/'results.json', dict(version=VERSION, dataset=dataset, seed=seed, training=training,
        native_temperature=temperature, output_members=100, native_dropout_forward_passes=10,
        marginal_representation='100 equal-probability midpoint quantiles of author Gaussian approximation',
        methods={name:acc.result() for name,acc in accumulators.items()}, fit_metadata=fit_meta,
        calibration_fit_seconds=fit_seconds, full_test_split=True, test_windows=windows, test_batches=steps,
        native_gaussian=dict(NLL=native['nll_sum']/native['n'], NLL_convention='author GaussianNLLLoss(full=False, eps=1e-6)', PICP95=native['interval_covered']/native['n'],
            MPIW95=native['interval_width_sum']/native['n']), total_test_seconds=time.monotonic()-start))
    print('COMPLETE', dataset, seed, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--dataset', required=True, choices=['PEMS03','PEMS04','PEMS07','PEMS08'])
    parser.add_argument('--seed', type=int, required=True); options = parser.parse_args()
    torch.set_num_threads(2); run(options.dataset, options.seed)
