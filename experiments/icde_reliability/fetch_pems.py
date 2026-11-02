"""Fetch native PeMS benchmark archives from the Torch Spatiotemporal maintainers."""
import json
import os
import time
import zipfile
from pathlib import Path
import requests
import numpy as np

ROOT = Path(os.environ.get('ICDE_RUN_ROOT', '/home/wyj/icde_reliability_20261102'))/'deepstuq/data'
ROOT.mkdir(parents=True, exist_ok=True)
expected = {'03': (26208, 358), '04': (16992, 307), '07': (28224, 883), '08': (17856, 170)}
records = []
for number, shape in expected.items():
    archive = ROOT/('pems'+number+'.zip')
    url = 'https://huggingface.co/datasets/TorchSpatiotemporal/ProcessedDatasets/resolve/main/traffic/'+archive.name+'?download=true'
    if not archive.exists():
        for attempt in range(3):
            try:
                temporary = archive.with_suffix('.part')
                with requests.get(url, stream=True, timeout=(20, 120)) as response:
                    response.raise_for_status()
                    with temporary.open('wb') as stream:
                        for chunk in response.iter_content(1024*1024):
                            stream.write(chunk)
                if not zipfile.is_zipfile(temporary):
                    raise ValueError('Downloaded file is not a ZIP archive')
                temporary.replace(archive); break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(5)
    destination = ROOT/('PEMS'+number); destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        candidates = [n for n in handle.namelist() if Path(n).name.lower() == 'pems'+number+'.npz']
        if len(candidates) != 1:
            raise ValueError((number, handle.namelist()))
        path = destination/('pems'+number+'.npz')
        path.write_bytes(handle.read(candidates[0]))
    array = np.load(path)['data']
    if array.shape[:2] != shape or not np.isfinite(array).all():
        raise ValueError((number, array.shape, shape))
    record = dict(dataset='PEMS'+number, source=url, shape=list(array.shape), bytes=path.stat().st_size)
    records.append(record)
    (ROOT/'dataset_records.json').write_text(json.dumps(records, indent=2))
    print('READY', record, flush=True)
