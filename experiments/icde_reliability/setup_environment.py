"""Install the tested GPU/runtime lock into the active Python environment."""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--requirements',type=Path,required=True)
args=parser.parse_args()
lines=[line for line in args.requirements.read_text().splitlines() if line and not line.startswith(('#','--'))]
gpu=[line for line in lines if line.lower().startswith(('torch==','torchvision==','torchaudio=='))]
if len(gpu)!=3:
    raise ValueError('Expected the three pinned PyTorch CUDA packages')
# Separate the CUDA wheel index so ordinary packages use the normal PyPI index.
# The second pass resolves every dependency and pip check verifies the result.
subprocess.run([sys.executable,'-m','pip','install','--no-deps','--index-url',
    'https://download.pytorch.org/whl/cu121',*gpu],check=True)
with tempfile.TemporaryDirectory(prefix='eab-requirements-') as directory:
    path=Path(directory)/'runtime.txt';path.write_text('\n'.join(lines)+'\n')
    subprocess.run([sys.executable,'-m','pip','install','-r',str(path)],check=True)
subprocess.run([sys.executable,'-m','pip','check'],check=True)
