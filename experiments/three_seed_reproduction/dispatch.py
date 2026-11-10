"""Share known small training jobs; retain one draining card for large jobs."""
LOW_DIM_K2={'etth1','etth2','ettm1','ettm2','exchange_ltsf','weather_ltsf','illness_ltsf'}
LOW_DIM_DTS={'ETTh1','ETTh2','ETTm1','ETTm2','ExchangeRate','ILI'}
PROFILED_DTS=LOW_DIM_DTS|{'Electricity','SolarEnergy','Traffic'}


def shareable(task):
    return ((task['backbone']=='K2VAE' and task['dataset'] in LOW_DIM_K2) or
            (task['backbone']=='DiffusionTS' and task['dataset'] in PROFILED_DTS))


def plan_dispatch(pending,occupants,gpus,stage,limit=3):
    queue=list(pending);owners={g:list(occupants.get(g,[])) for g in gpus};plans=[]
    drain=None
    if stage=='training' and queue and not shareable(queue[0]):
        candidates=[g for g in gpus if owners[g] and all(shareable(t) for t in owners[g])]
        # Retain a small-dataset drain before using a larger DTS dataset card.
        drain=min(candidates,key=lambda g:(
            any(t['backbone']=='DiffusionTS' and t['dataset'] not in LOW_DIM_DTS for t in owners[g]),
            len(owners[g]))) if candidates else None
    for gpu in gpus:
        while queue:
            here=owners[gpu]
            if not here:index=0
            elif stage!='training' or len(here)>=limit or not all(shareable(t) for t in here) or gpu==drain:break
            else:
                index=next((i for i,t in enumerate(queue) if shareable(t)),None)
                if index is None:break
            task=queue.pop(index);plans.append((gpu,task));here.append(task)
            if stage!='training' or not shareable(task):break
    return plans,queue
