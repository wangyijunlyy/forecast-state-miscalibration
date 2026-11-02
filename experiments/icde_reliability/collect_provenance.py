"""Preserve the actual server source/configuration dependencies privately.

This snapshot is for reproduction and source comparison. Public redistribution
must follow the recorded source licenses; third-party code with no license is
fetched by revision in the eventual public artifact instead.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import tarfile
import urllib.parse
from pathlib import Path

ROOTS = {
    'k2vae_training': '/home/wyj/K2VAE',
    'k2vae_evaluation': '/home/wyj/SPARC/K2VAE_dacu_k2vae_ltsf',
    'ppm': '/home/wyj/SPARC/PPM',
    'nsdiff_tmdm': '/home/wyj/SPARC/NsDiff',
    'diffusionts': '/home/wyj/SPARC/NsDiff_DiffusionTS_SPARC_PIT_20260716',
    'timediff': '/home/wyj/SPARC/NsDiff_TimeDiff_SPARC_PIT_Fresh',
    'native_training_interfaces': '/home/wyj/three_seed_reproduction_20261031/code',
    'diffusion_evaluation_interfaces': '/home/wyj/distribution_binned_diffusion/code',
    'historical_source_filter': '/home/wyj/source_filtered_reproduction/code',
    'icde_experiment': '/home/wyj/icde_reliability_20261102/code',
    'torch_timeseries': '/home/wyj/miniconda3/envs/k2vae/lib/python3.10/site-packages/torch_timeseries',
    'deepstuq_author': '/home/wyj/icde_reliability_20261102/third_party/DeepSTUQ_Pytorch',
}
SUFFIXES = {'.py', '.yaml', '.yml', '.conf', '.toml', '.ini', '.cfg', '.sh'}
TOP_EXCLUDED = {'data', 'datasets', 'results', 'logs', 'outputs', 'checkpoints', 'lightning_logs', 'wandb',
                'archive', 'archives', 'venv', '.venv', 'env', 'node_modules'}
EVERYWHERE_EXCLUDED = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache', '.ipynb_checkpoints'}
TOP_SOURCE_DIRECTORIES = {'config', 'configs', 'probts', 'scripts', 'sparc_adapters', 'src', 'utils',
    'model_ppm', 'sparc_ppm', 'model', 'models', 'layers', 'nn', 'exp', 'exps', 'docs', 'tests',
    'lib', 'data_provider', 'external', 'datasets', 'dataloader', 'scaler', 'losses', 'optim', 'metrics',
    'dataset', 'core', 'constants', 'utils', 'augmentation', 'sampler'}


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def source_files(root):
    for directory, directories, files in os.walk(root, followlinks=False):
        relative = Path(directory).relative_to(root)
        excluded = EVERYWHERE_EXCLUDED | (TOP_EXCLUDED if relative == Path('.') and root.name != 'torch_timeseries' else set())
        directories[:] = sorted(d for d in directories if d not in excluded)
        if relative == Path('.') and root.name not in ('code', 'torch_timeseries'):
            directories[:] = [d for d in directories if d in TOP_SOURCE_DIRECTORIES]
        for name in sorted(files):
            path = Path(directory)/name
            special = name.lower().startswith(('license', 'copying', 'readme', 'requirements', 'environment'))
            modifications = name.lower() == 'modifications.md'
            if path.is_file() and not path.is_symlink() and (path.suffix in SUFFIXES or special or modifications):
                yield path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    records = []
    archive_path = args.output/'server_source_snapshot.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        for name, source in ROOTS.items():
            root = Path(source)
            if not root.is_dir():
                raise FileNotFoundError(source)
            files = list(source_files(root)); licenses = []
            for path in files:
                relative = path.relative_to(root)
                archive.add(path, arcname=str(Path(name)/relative), recursive=False)
                if path.name.lower().startswith(('license', 'copying')):
                    licenses.append(str(relative))
            remote = run(['git', '-C', source, 'remote', 'get-url', 'origin'])
            if remote and '://' in remote:
                url = urllib.parse.urlsplit(remote)
                remote = urllib.parse.urlunsplit((url.scheme, url.hostname or '', url.path, '', ''))
            changed = run(['git', '-C', source, 'diff', '--name-only', 'HEAD'])
            records.append(dict(alias=name, source_path=source, source_files=len(files),
                source_bytes=sum(p.stat().st_size for p in files), source_license_files=licenses,
                git_revision=run(['git', '-C', source, 'rev-parse', 'HEAD']), git_origin=remote,
                modified_tracked_source_files=[p for p in (changed or '').splitlines() if Path(p).suffix in SUFFIXES]))
    environment = dict(python=sys.version, platform=platform.platform(), cpu_logical_processors=os.cpu_count(),
        memory_total_kib=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:'))),
        packages=json.loads(run([sys.executable, '-m', 'pip', 'list', '--format=json'])),
        gpu=run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']))
    (args.output/'source_provenance.json').write_text(json.dumps(records, indent=2)+'\n')
    (args.output/'environment.json').write_text(json.dumps(environment, indent=2)+'\n')
    print(json.dumps(dict(source_roots=len(records), source_files=sum(r['source_files'] for r in records),
                          archive_bytes=archive_path.stat().st_size,
                          roots=[dict(alias=r['alias'], source_files=r['source_files'], licenses=r['source_license_files']) for r in records]), indent=2))


if __name__ == '__main__':
    main()
