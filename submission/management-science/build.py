#!/usr/bin/env python3
"""Compile the review pair with the existing Tectonic cache; never fetch TeX."""
from pathlib import Path
import re
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parent
build = root / 'build'
build.mkdir(exist_ok=True)
engine = shutil.which('tectonic')
if not engine:
    raise SystemExit('Existing tectonic executable required; do not auto-install.')
for name in ('manuscript', 'companion'):
    if name == 'companion':
        labels = [line for line in (build / 'manuscript.aux').read_text().splitlines()
                  if line.startswith('\\newlabel')]
        (root / 'main-labels.tex').write_text(
            '% Generated from the journal main manuscript; refresh with build.py.\n'
            + '\n'.join(labels) + '\n')
    result = subprocess.run([engine, '--only-cached', '--untrusted', '--keep-logs',
                             '--keep-intermediates', '--outdir', str(build), name + '.tex'],
                            cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (build / (name + '-console.log')).write_text(result.stdout)
    if result.returncode:
        print(result.stdout[-5000:]); raise SystemExit(result.returncode)
    log = (build / (name + '.log')).read_text()
    if re.search(r'(undefined references|Citation .+ undefined|Reference .+ undefined|! LaTeX Error)', log):
        raise SystemExit(name + ': unresolved reference or TeX error; inspect build log.')
    shutil.copyfile(build / (name + '.pdf'), root / (name + '.pdf'))
    boxes = re.findall(r'Overfull \\[hv]box.*', log)
    print(name + ': compiled; overfull boxes: ' + str(len(boxes)))
    for box in boxes: print('  ' + box)
