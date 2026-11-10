"""Requirement-level evidence audit and a text-only result report."""
import json,math
from collections import Counter,defaultdict
from pathlib import Path
from core import severe_trigger
from validation_runtime import DTS_VALIDATION_VERSION
METHODS={'Native','DC16','DC64'}
METRICS=('CRPS','QICE','QICE_percent','MSE','MAE','RMSE','MedianMAE')
EXPECTED={'K2VAE':108,'PPM':21,'NsDiff':27,'TMDM':27,'DiffusionTS':27,'TimeDiff':27}


def evidence_audit(root,tasks):
    issues=[];completed=0;training_count=0;validation_count=0;expected_gates=defaultdict(lambda:{'16':False,'64':False})
    scope=defaultdict(set)
    for t in tasks:scope[t['backbone'],t['dataset'],t['horizon']].add(t['seed'])
    if len(tasks)!=237 or len(scope)!=79 or Counter(t['backbone'] for t in tasks)!=EXPECTED:issues.append('Manifest does not cover237 required runs and79 configurations')
    if any(v!={1,2,3} for v in scope.values()):issues.append('Missing independent training seed in task scope')
    if len({t['id'] for t in tasks})!=len(tasks):issues.append('Duplicate task identity')
    gp=root/'frozen_gates.json';gates=json.loads(gp.read_text()) if gp.exists() else None
    if not gates or gates.get('validation_complete_tasks')!=237 or not gates.get('no_test_metrics_used'):issues.append('Missing complete validation gate freeze')
    for t in tasks:
        tid=t['id'];out=Path(t['run_root']);files={name:out/(name+'.json') for name in ('training','validation','results')}
        if any(not p.exists() for p in files.values()):
            issues.append(tid+': missing '+','.join(k for k,p in files.items() if not p.exists()));continue
        tr,val,r=(json.loads(files[name].read_text()) for name in ('training','validation','results'))
        ck=Path(tr.get('checkpoint',''))
        if not tr.get('training_complete') or not ck.is_file() or not ck.stat().st_size:issues.append(tid+': missing training checkpoint')
        else:
            training_count+=1
            if not ck.resolve().is_relative_to(out.resolve()):issues.append(tid+': checkpoint outside fresh task output')
        for key in ('task_id','seed','backbone','dataset','horizon'):
            if tr.get(key)!=(tid if key=='task_id' else t[key]):issues.append(tid+': training identity mismatch '+key)
        if tr.get('metadata',{}).get('training_batch_size')!=t['training_batch_size']:issues.append(tid+': training batch mismatch')
        if t['backbone']=='DiffusionTS' and tr.get('metadata',{}).get('validation_protocol')!=DTS_VALIDATION_VERSION:issues.append(tid+': obsolete DiffusionTS validation protocol')
        for name in (['cond_pred_model.pth','cond_pred_model_g.pth'] if t['backbone']=='NsDiff' else ['cond_pred_model.pth'] if t['backbone']=='TMDM' else []):
            p=ck.parent/name
            if not p.is_file() or not p.stat().st_size:issues.append(tid+': missing component '+name)
        if val.get('task_id')!=tid or val.get('seed')!=t['seed'] or not val.get('validation_complete') or not val.get('full_validation_split'):issues.append(tid+': incomplete/mismatched validation')
        else:validation_count+=1
        fit_path=out/'fit_metadata.json'
        if not fit_path.exists() or not (out/'calibrators.pkl').is_file():issues.append(tid+': missing fitted calibrator evidence')
        else:
            fit=json.loads(fit_path.read_text())
            if fit.get('fit_split')!='train' or fit.get('seed')!=t['seed']+10000 or fit.get('output_samples')!=100:issues.append(tid+': calibration data/seed/sample configuration mismatch')
            for bins,nh in [('16',1),('64',4)]:
                cfg=fit.get('configurations',{}).get(bins,{})
                expected=dict(horizon_length=t['horizon'],n_mean_bins=4,n_scale_bins=4,n_horizon_bins=nh,min_bin_size=128,max_pool_size=20000)
                if cfg!=expected:issues.append(tid+': fitted bin configuration mismatch '+bins)
        groupkey=t['backbone']+'/'+t['dataset']
        for bins in ('16','64'):
            d=val.get('diagnostics',{}).get(bins)
            if not d:issues.append(tid+': missing validation diagnostic '+bins);continue
            expected=severe_trigger(d.get('R_val') or 0,d.get('A_val',0),d.get('binned_nonfinite',False))
            if d.get('triggered')!=expected:issues.append(tid+': MAD decision mismatch '+bins)
            expected_gates[groupkey][bins]=expected_gates[groupkey][bins] or expected
        if r.get('task',{}).get('id')!=tid or r.get('task',{}).get('seed')!=t['seed']:issues.append(tid+': result identity mismatch')
        if not r.get('full_test_split') or r.get('test_windows',0)<=0 or r.get('test_batches',0)<=0:issues.append(tid+': missing full-test traversal')
        if r.get('test_seed')!=t['seed']+30000:issues.append(tid+': test sampling seed mismatch')
        expected_ck=ck.parent if t['backbone'] in ('NsDiff','TMDM','DiffusionTS','TimeDiff') else ck
        if Path(r.get('checkpoint','')).resolve()!=expected_ck.resolve():issues.append(tid+': evaluated checkpoint mismatch')
        common=r.get('common_normalized',{});native=r.get('native_metrics',{})
        if set(common)!=METHODS or set(native)!=METHODS:issues.append(tid+': missing method scores');continue
        counts={common[n].get('coordinate_count',0) for n in METHODS}
        if len(counts)!=1 or min(counts)<=0:issues.append(tid+': coordinate_count mismatch across methods')
        for name in METHODS:
            for metric in METRICS:
                x=common[name].get(metric)
                if x is None or not math.isfinite(x):issues.append(tid+': missing/nonfinite common metric '+name+'/'+metric)
            if not math.isfinite(native[name].get('CRPS',float('nan'))):issues.append(tid+': missing/nonfinite native CRPS')
        if gates:
            if r.get('mad_trigger')!=gates.get('groups',{}).get(groupkey):issues.append(tid+': test uses different MAD switch')
            if files['results'].stat().st_mtime<gp.stat().st_mtime:issues.append(tid+': result predates gate freeze')
        completed+=1
    if gates and dict(expected_gates)!=gates.get('groups'):issues.append('Frozen dataset/bin gates do not match complete validation evidence')
    return dict(expected_tasks=237,training_complete_tasks=training_count,validation_complete_tasks=validation_count,
                complete_test_tasks=completed,issues=issues,complete=not issues and completed==237)


def render_report(aggregates,audit,gates,native_aggregates=None):
    complete=audit.get('complete',False);count=audit.get('complete_test_tasks',0)
    text=['# 三训练种子：Native、16箱与64箱对比','',f'完整测试结果：{count}/237。'+('全部完成。' if complete else '未完成，以下内容不能作为最终对比结果。'),'',
          '每个配置使用3个独立训练种子。模型级汇总先在各种子内等权平均数据集／预测长度任务，再跨种子计算均值和样本标准差（ddof=1）。',
          '下表采用标准化空间的共同经验CRPS、10区间QICE和预测均值的MSE、MAE、RMSE；各指标均越小越好。QICE以百分点显示。','']
    if complete:
        text+=['| 模型 | 方法 | CRPS | QICE（百分点） | MSE | MAE | RMSE |','|---|---|---:|---:|---:|---:|---:|']
        for row in aggregates:
            cells=[f'{row[m+"_mean"]:.6g} ± {row[m+"_std"]:.3g}' for m in ('CRPS','QICE_percent','MSE','MAE','RMSE')]
            text.append('| '+row['backbone']+' | '+row['method']+' | '+' | '.join(cells)+' |')
    if complete and native_aggregates:
        text+=['','原骨干口径汇总用于与历史结果对照。K2VAE保留原加权分位数CRPS，其他模型保留各自原生CRPS与QICE定义；共同口径的比较见上表。','',
               '| 模型 | 方法 | 原生CRPS | 原生QICE（百分点） |','|---|---|---:|---:|']
        for row in native_aggregates:
            cells=[f'{row[m+"_mean"]:.6g} ± {row[m+"_std"]:.3g}' for m in ('CRPS','QICE_percent')]
            text.append('| '+row['backbone']+' | '+row['method']+' | '+' | '.join(cells)+' |')
    if gates:
        text+=['','MAD开关仅由完整验证集诊断决定：未滤波校准的CRPS至少恶化5倍且幅度放大至少100倍，或出现非有限输出时触发。模型／数据集／箱数分别确定；同组任一训练种子或预测长度严重失稳即开启，测试前冻结。','',
               '| 模型／数据集 | 16箱MAD | 64箱MAD |','|---|---|---|']
        for key,value in sorted(gates['groups'].items()):text.append(f'| {key} | '+('开启' if value['16'] else '关闭')+' | '+('开启' if value['64'] else '关闭')+' |')
    text+=['','逐配置均值和标准差：`task_mean_std.csv`；逐种子原始分数：`raw_scores.csv`；配对变化：`paired_changes_mean_std.csv`。',
           '原骨干指标口径另外保存在`native_raw_scores.csv`和`native_task_mean_std.csv`、`native_backbone_mean_std.csv`，用于与历史结果对照。','']
    return '\n'.join(text)
