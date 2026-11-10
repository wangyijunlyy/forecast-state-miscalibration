"""Aggregate complete diffusion-cohort results and historical Table1 references."""
import argparse,csv,json,math
from pathlib import Path
from statistics import mean

FILES={'NsDiff':'nsdiff_sparc_pit_3seed_task_means.csv','TMDM':'tmdm_sparc_pit_3seed_task_means.csv',
'DiffusionTS':'diffusionts_sparc_pit_3seed_task_means.csv','TimeDiff':'timediff_sparc_pit_task_means.csv'}
ALIASES={'ECL':'Electricity','EXG':'ExchangeRate','Solar':'SolarEnergy'}


def write_csv(p,rows):
    if rows:
        with p.open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=Path(__file__).with_name('reference'))
    a=p.parse_args();refs={}
    for b,file in FILES.items():
        for r in csv.DictReader((a.reference/file).open()):refs[b,ALIASES.get(r['dataset'],r['dataset'])]=r
    table={r['backbone']:r for r in csv.DictReader((a.reference/'main_table.csv').open())}
    manifest=json.loads((a.root/'manifest.json').read_text());expected={t['id']:t for t in manifest}
    rows=[];raw=[]
    for p in sorted(a.root.glob('*/results.json')):
        r=json.loads(p.read_text());t=r['task'];assert t==expected[t['id']]
        assert r['full_test_split'] and r['fit']['fit_split']=='train'
        assert r['fit']['fit_windows']==min(100,len(r['fit']['selected_train_indices']))
        assert len(set(r['fit']['selected_train_indices']))==r['fit']['fit_windows']
        assert r['fit']['residual_filter']['multiplier']==5
        scores=r['metrics'];ref=refs[t['backbone'],t['dataset']]
        for v in scores.values():assert all(math.isfinite(v[k]) for k in ['CRPS','QICE_percent'])
        assert len({v['coordinate_count'] for v in scores.values()})==1
        f=r['fit']['residual_filter']
        row=dict(id=t['id'],backbone=t['backbone'],dataset=t['dataset'],horizon=t['horizon'],backbone_seed=1,
            checkpoint=t['checkpoint'],checkpoint_selection=t['checkpoint_selection'],
            sampling_chunk_size=t['sampling_chunk_size'],test_windows=r['test_windows'],
            calibration_windows=r['fit']['fit_windows'],calibration_points=r['fit']['calibration_points_retained'],
            clipped_percent=100*(f['clipped_lower']+f['clipped_upper'])/f['total_points'])
        for label,k in [('native','Native'),('binned','Binned'),('filtered','BinnedFiltered')]:
            row[label+'_crps']=scores[k]['CRPS'];row[label+'_qice_percent']=scores[k]['QICE_percent']
        row.update(historical_native_crps=float(ref['baseline_crps']),historical_native_qice_percent=100*float(ref['baseline_qice']),
            historical_sparc_crps=float(ref['sparc_pit_crps']),historical_sparc_qice_percent=100*float(ref['sparc_pit_qice']),
            native_crps_drift_vs_historical=scores['Native']['CRPS']-float(ref['baseline_crps']),
            total_seconds=r['total_seconds'],test_cache_hits=r['test_cache_hits'],standard_deviation='TBD')
        rows.append(row);raw.append(r)
    write_csv(a.root/'task_comparison.csv',rows)
    summaries=[]
    keys=['native_crps','binned_crps','filtered_crps','native_qice_percent','binned_qice_percent','filtered_qice_percent']
    for b in FILES:
        rr=[r for r in rows if r['backbone']==b]
        if len(rr)!=9:continue
        assert len({r['dataset'] for r in rr})==9
        h=table[b]
        summaries.append(dict(backbone=b,complete_tasks=9,backbone_seed=1,**{k:mean(r[k] for r in rr) for k in keys},
            table1_native_crps=float(h['baseline_crps']),table1_sparc_crps=float(h['sparc_pit_crps']),
            table1_sparc_qice_percent=100*float(h['sparc_pit_qice']),
            binned_crps_wins_vs_current_native=sum(r['binned_crps']<r['native_crps'] for r in rr),
            filtered_crps_wins_vs_current_native=sum(r['filtered_crps']<r['native_crps'] for r in rr),
            native_crps_mean_abs_drift_vs_historical=mean(abs(r['native_crps_drift_vs_historical']) for r in rr),
            standard_deviation='TBD'))
    write_csv(a.root/'aggregate_comparison.csv',summaries)
    report=['# 四个扩散骨干：binned 原版与 MAD 滤波版','',
        f'已完成 {len(rows)}/36 项完整测试。四个骨干各9个任务，使用现有 seed1 检查点。'
        '这不是表1三种子结果的补齐；表1列为历史汇总。全部标准差为 TBD。','',
        '每项使用同一批 Native 样本评价原版 binned 和 MAD 滤波版，均不结合 SPARC。'
        '校准固定使用100个不同训练窗口（不足100则全部）、每窗口100个样本、最多50,000个标量坐标；'
        'horizon/mean/log-scale 各4箱。MAD 倍数固定为5。','',
        '测试完整遍历原骨干的 fast/non-overlapping 测试集，不截取前若干批次。'
        'CRPS 保留原经验样本评分定义，QICE 保留原十箱及两端越界并入首末箱的规则，以下以百分比显示。','',
        '| 骨干 | Native CRPS | binned CRPS | 滤波 binned CRPS | 表1 SPARC-PIT CRPS | binned QICE (%) | 滤波 QICE (%) | 表1 SPARC-PIT QICE (%) | 标准差 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for s in summaries:
        kk=['native_crps','binned_crps','filtered_crps','table1_sparc_crps','binned_qice_percent','filtered_qice_percent','table1_sparc_qice_percent']
        report.append('| '+' | '.join([s['backbone']]+[f'{s[k]:.6f}' for k in kk]+['TBD'])+' |')
    report+=['','| 骨干 | 数据集 | Native CRPS | binned CRPS | 滤波 binned CRPS | 历史 SPARC-PIT CRPS | binned QICE (%) | 滤波 QICE (%) | 测试窗口数 | 标准差 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        kk=['native_crps','binned_crps','filtered_crps','historical_sparc_crps','binned_qice_percent','filtered_qice_percent']
        report.append('| '+' | '.join([r['backbone'],r['dataset']]+[f'{r[k]:.6f}' for k in kk]+[str(r['test_windows']),'TBD'])+' |')
    report+=['','## 与历史结果的可比范围','',
        '新 Native 是与两种 binned 同样本的直接对照；历史 SPARC-PIT 未在本次样本上重跑。'
        '历史表1为三种子或用户提供的任务均值，本次仅有 seed1，因此不能把差值解释为严格配对改进。'
        '本轮评估批次和采样分块已记录在清单中，并非历史运行的逐点重放；'
        'DiffusionTS 的可微 infill 优化也可能受批次／分块数值设置影响。'
        '校准器效果应优先比较本轮同配置、同样本的 Native 对照。'
        '逐任务 CSV 同时给出新旧 Native 差值，避免把检查点或采样差异归因于校准器。','',
        'NsDiff–Electricity 的差异需单独注意：本次 Native CRPS 约0.28608，'
        '旧归档为0.52328。使用原生 NsDiffForecast 评估入口、原 batch_size=32 和'
        'minisample=1，对当前保留检查点重新遍历完整测试集，得到0.28610，'
        '与本次 Native 一致。旧记录与当前检查点的对应关系仍需另行核对，'
        '不能把这一部分 Native 差异算作 binned 的收益。复核数据见 '
        '`../validation/native_interface_audit.json`。','',
        'DiffusionTS 的 ETTh1、ETTm2、ILI 使用保留的旧归档检查点，其他六项使用当前保留检查点。'
        '训练缓存不复用；TimeDiff 测试缓存仅在检查点和确定性测试范围核对一致时只读复用。'
        'TimeDiff 的 Electricity、SolarEnergy、Traffic 和 ExchangeRate 通过原模型 forward '
        '批量执行独立轨迹；相同初始噪声的直接核对误差约1e-6。其他任务保留原采样路径。'
        '所用检查点、保存参数、训练窗口索引和计算分块大小均保存在 manifest / results.json。','',
        '每个种子内对9个任务等权平均；不跨骨干混合绝对指标。完整结果见 '
        '`task_comparison.csv`、`aggregate_comparison.csv`、`*/results.json` 与 `*/run.log`。'
        '原论文表格未自动修改。']
    (a.root/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(dict(completed=len(rows),summaries=summaries),indent=2))


if __name__=='__main__':main()
