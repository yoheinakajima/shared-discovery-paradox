#!/usr/bin/env python3
"""Validate saved artifacts only. No experiment, network, provider or state write."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

here = Path(__file__).resolve().parent
repo = here.parents[1]
manifest = json.loads((here / 'provenance.json').read_text())
for item in manifest['preserved_baseline_files'] + manifest['prepared_files']:
    path = repo / item['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], str(path)
tex = (here / 'manuscript.tex').read_text()
abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex, re.S).group(1)
assert len(abstract.split()) <= 250
assert '\\setstretch{1.5}' in tex and '\\documentclass[11pt]' in tex
assert '\\author{}' in tex and '\\date{}' in tex
assert not any(x in tex for x in ['\\thanks{', 'Untapped Capital', 'github.com/yoheinakajima'])
keys = set()
for group in re.findall(r'\\cite[pt]?\{([^}]+)\}', tex):
    keys.update(group.split(','))
bibkeys = set(re.findall(r'@\w+\{([^,]+),', (here / 'bibliography.bib').read_text()))
assert keys <= bibkeys, keys - bibkeys
assert keys == set(json.loads((here / 'reference-metadata.json').read_text()))
for name in ('manuscript', 'companion'):
    pdf = here / (name + '.pdf')
    text = subprocess.check_output(['pdftotext', '-layout', str(pdf), '-']).decode()
    info = subprocess.check_output(['pdfinfo', str(pdf)]).decode()
    assert '??' not in text
    assert not re.search(r'^Author:\s*\S', info, re.M)
    if name == 'companion':
        assert 'References' not in text
    assert 'Untapped Capital' not in text
    assert not any(identity in text for identity in (
        'yohei@untapped.vc', '0009-0009-1293-6053', 'Bellevue, Washington',
        'supported by Untapped', 'founder and General Partner'))
    print(name, re.search(r'Pages:\s*(\d+)', info).group(1) + ' pages; metadata/text checks passed')
named_pdf = here / 'author-information.pdf'
named_text = subprocess.check_output(['pdftotext', '-layout', str(named_pdf), '-']).decode()
named_info = subprocess.check_output(['pdfinfo', str(named_pdf)]).decode()
assert re.search(r'^Author:\s*Yohei Nakajima\s*$', named_info, re.M)
assert re.search(r'Pages:\s*1\s*$', named_info, re.M)
for value in ('Yohei Nakajima', 'Untapped Capital', 'yohei@untapped.vc',
              '0009-0009-1293-6053', 'Bellevue, Washington, USA'):
    assert value in named_text, value
orcid = json.loads((here / 'tool-use-evidence.json').read_text())['orcid']['id'].replace('-', '')
total = 0
for digit in orcid[:-1]:
    total = (total + int(digit)) * 2
checksum = (12 - total % 11) % 11
assert ('X' if checksum == 10 else str(checksum)) == orcid[-1]
print('Named author asset: 1 page; supplied metadata and ORCID checksum passed; blind files separate')
tasks = json.loads((here / 'task-disposition.json').read_text())['tasks']
assert len(tasks) == 28 and len({t['id'] for t in tasks}) == 28
receipt = json.loads((here / 'verification-receipt.json').read_text())
assert receipt['exit_code'] == 0 and receipt['all_original_assertions_passed']
print('PASS: hashes, format, 28 citations, 28 task dispositions and saved verification receipt.')
print('This is not a proof certification or a readiness/owner-attestation check.')
