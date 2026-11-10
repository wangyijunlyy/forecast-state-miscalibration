"""Readable evidence report from complete runs; no manuscript modifications."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

METRICS=['MSE','CRPS','QICE','StateError','BrierMean']
DATASETS=['ETTh1','ETTh2','ETTm1','ETTm2','Electricity','ExchangeRate','Traffic','ILI','SolarEnergy']
LABELS={'FSPCPublished':'现有 FSPC','FSPCMean':'仅均值修改','FSPCScale':'仅尺度修改','FSPCJoint':'联合修改',
    'FSPCRefit':'FSPC','Qian2025SoftCRPS':'Learned PIT','DistributionRecalibration2022':'Distribution Recalibration','QRF2016':'QRF'}


def metrics(v):
    state=v['state_coverage']['state_error']
    return dict(MSE=v['MSE'],CRPS=v['CRPS'],QICE=v['QICE_percent'],
        StateError=None if state is None else 100*state,BrierMean=float(np.mean(v['events']['brier'])))


def report(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text());by_mode=defaultdict(list)
    for job in manifest:
        p=root/job['mode']/'runs'/job['task']['id']/'results.json'
        if p.exists():
            d=json.loads(p.read_text());assert d['full_test_split'];by_mode[job['mode']].append(d)
    complete=sum(map(len,by_mode.values()))
    lines=['# Diffusion-TS 位置与尺度修正实验', '',
        f'已完成 **{complete}/{len(manifest)}** 个完整测试任务。以下只汇总同一数据集三次运行全部完成的结果；缺失任务见 `analysis/coverage.csv`。', '',
        '本报告为方法诊断，正文与已有主结果未改动。原始采样与关闭截断的采样分别呈现。', '',
        '## 实验设置', '',
        '- 保留冻结的原生模型；输出均为 100 个有序边际值。所有候选参数只使用历史训练和验证数据。',
        '- QICE 与 StateError 的单位为百分点（pp）；五项指标均为越小越好。',
        '- 按验证窗口时间顺序划分拟合与选择段，同一窗口的变量和预测步不拆分；预测目标重叠的窗口被剔除。',
        '- 位置候选采用全局仿射均值加收缩后的状态偏差；尺度候选包含非负独立方差截距和状态残差形状。',
        '- 在验证选择段上最小化 CRPS，约束 MSE 不超过使用同一划分的 FSPC；选择权重冻结后在完整校准样本上重新拟合。',
        '- 同时保留未重新拟合的 blocked 版本，供纯消融检查；原始 FSPC 的完整样本版本作为主参照。',
        '- ILI 只有一个原协议验证窗口，单列采用“训练拟合、验证选择”的设置，不用于声称验证时间块协议的普适性。',
        '- 关闭截断的任务重新生成训练与验证集合，并完整重新拟合 EMOS、Learned PIT、Distribution Recalibration、QRF；阈值和诊断状态沿用原协议。',
        '- 完整缓存复用仅限保留坐标数等于完整测试坐标数的任务，并核对 Native/FSPC 数值。大型任务仍遍历完整原生测试迭代器。', '',
        '## 原始采样：四组消融', '']
    grouped=defaultdict(list)
    for d in by_mode['original']:grouped[d['task']['dataset']].append(d)
    methods=['FSPCPublished','FSPCMean','FSPCScale','FSPCJoint']
    for metric in METRICS:
        lines.extend([f'### {metric} ↓', '', '| 数据集 | '+' | '.join(LABELS[n] for n in methods)+' |',
            '|---|'+'---:|'*len(methods)])
        for dataset in DATASETS:
            ds=grouped[dataset]
            if len(ds)!=3:continue
            values=[[metrics(d['methods'][method])[metric] for d in ds] for method in methods]
            means=[None if any(v is None for v in a) else float(np.mean(a)) for a in values]
            best=min([x for x in means if x is not None],default=None);cells=[]
            for mean,a in zip(means,values):
                if mean is None:cells.append('—');continue
                text=f'{mean:.4f}<sub>±{np.std(a,ddof=1):.4f}</sub>'
                cells.append('**'+text+'**' if abs(mean-best)<1e-10 else text)
            lines.append('| '+dataset+' | '+' | '.join(cells)+' |')
        lines.append('')
    complete_datasets=[x for x in DATASETS if len(grouped[x])==3]
    if len(complete_datasets)==9:
        lines.extend(['### 九个数据集的总体平均', '', '| 方法 | '+' | '.join(m+' ↓' for m in METRICS)+' |', '|---|'+'---:|'*len(METRICS)])
        for method in methods:
            cells=[]
            for metric in METRICS:
                dataset_values=[]
                for dataset in DATASETS:
                    a=[metrics(d['methods'][method])[metric] for d in grouped[dataset]]
                    if not any(v is None for v in a):dataset_values.append(np.mean(a))
                cells.append(f'{np.mean(dataset_values):.5f}')
            lines.append('| '+LABELS[method]+' | '+' | '.join(cells)+' |')
        lines.append('')
        baseline_root=root.parent/'external_baselines'
        external=[]
        for d in by_mode['original']:
            p=baseline_root/'runs'/d['task']['id']/'results.json'
            if not p.exists():break
            source=json.loads(p.read_text())
            np.testing.assert_allclose(source['methods']['Native']['CRPS'],d['methods']['Native']['CRPS'],atol=1e-9)
            external.append(source)
        if len(external)==27:
            comparison=['Native','FSPCPublished','FSPCJoint','EMOS','Qian2025SoftCRPS','DistributionRecalibration2022','QRF2016']
            lines.extend(['## 原始采样：完整候选与其他校准方法', '',
                '所有方法使用相同的九个完整测试集。标准差来自三次运行的 backbone 平均值；StateError 排除无合格状态的 ILI。联合修改为候选配置，未替换论文主方法。', '',
                '| 方法 | '+' | '.join(m+' ↓' for m in METRICS)+' |','|---|'+'---:|'*len(METRICS)])
            aggregated={}
            for method in comparison:
                per_seed=defaultdict(lambda:defaultdict(list))
                for d,old in zip(by_mode['original'],external):
                    record=d if method in ['FSPCPublished','FSPCJoint'] else old
                    for metric,value in metrics(record['methods'][method]).items():
                        if value is not None:per_seed[d['task']['seed']][metric].append(value)
                aggregated[method]={m:[np.mean(v[m]) for v in per_seed.values()] for m in METRICS}
            for method in comparison:
                cells=[]
                for metric in METRICS:
                    a=aggregated[method][metric];best=min(np.mean(aggregated[n][metric]) for n in comparison)
                    cell=f'{np.mean(a):.4f}<sub>±{np.std(a,ddof=1):.4f}</sub>'
                    cells.append('**'+cell+'**' if abs(np.mean(a)-best)<1e-10 else cell)
                lines.append('| '+LABELS.get(method,method)+' | '+' | '.join(cells)+' |')
            lines.append('')
    lines.extend(['## 关闭截断：与其他校准方法的比较', ''])
    grouped_new=defaultdict(list)
    for d in by_mode['unclipped']:grouped_new[d['task']['dataset']].append(d)
    comparison=['Native','FSPCRefit','EMOS','Qian2025SoftCRPS','DistributionRecalibration2022','QRF2016']
    for dataset in DATASETS:
        ds=grouped_new[dataset]
        if len(ds)!=3:continue
        original=grouped[dataset]
        old_collapse=np.mean([d['native_collapsed_fraction'] for d in original]) if len(original)==3 else None
        new_collapse=np.mean([d['native_collapsed_fraction'] for d in ds])
        lines.extend([f'### {dataset}', '',f'原生集合坍缩比例：原始采样 {100*old_collapse:.2f}%，关闭截断后 {100*new_collapse:.2f}%。' if old_collapse is not None else f'关闭截断后坍缩比例 {100*new_collapse:.2f}%。', '',
            '| 方法 | '+' | '.join(m+' ↓' for m in METRICS)+' |','|---|'+'---:|'*len(METRICS)])
        values={m:{metric:[metrics(d['methods'][m])[metric] for d in ds] for metric in METRICS} for m in comparison}
        for method in comparison:
            cells=[]
            for metric in METRICS:
                a=values[method][metric]
                if any(v is None for v in a):cells.append('—');continue
                best=min(np.mean(values[m][metric]) for m in comparison)
                text=f'{np.mean(a):.4f}<sub>±{np.std(a,ddof=1):.4f}</sub>'
                cells.append('**'+text+'**' if abs(np.mean(a)-best)<1e-10 else text)
            lines.append('| '+LABELS.get(method,method)+' | '+' | '.join(cells)+' |')
        lines.append('')
    lines.extend(['## 结论边界', '',
        '验证集上的 MSE 约束不构成测试集保证；重新拟合后的模型与未重新拟合版本分别保存在原始记录中。联合修改不保证五项指标同时改善。跨数据集平均不能替代逐数据集失败分析。', '',
        '尺度模块还包含稳定残差标准化和对原先坍缩来源的利用，因此本消融测量整个尺度修正模块，不能把全部变化单独归因于方差截距。', '',
        '原始逐任务五项指标、均值与标准差、缺失任务与失败历史见 `analysis/`。所有原始记录保留在 `original/runs/` 与 `unclipped/runs/`。', ''])
    (root/'README.md').write_text('\n'.join(lines))
    print('Report',root/'README.md','complete',complete,'/',len(manifest))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);report(p.parse_args().root)
