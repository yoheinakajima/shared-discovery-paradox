#!/usr/bin/env python3
"""Compile the review pair or separate named page with cached Tectonic; never fetch TeX."""
from pathlib import Path
import argparse
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
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--author-information-only', action='store_true',
                    help='Build only the separate named author asset, not blind review files.')
args = parser.parse_args()
names = ('author-information',) if args.author_information_only else ('manuscript', 'companion')
for name in names:
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
