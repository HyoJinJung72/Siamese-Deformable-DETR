"""Collect the Phase 1 alignment-robustness sweep into a table.

Reads the COCO summary lines from each run's eval.log and prints AP50 / AP50-95
as a method x delta grid, plus the drop relative to delta=0.
"""
import re
import sys
from pathlib import Path

METHODS = ['single', 'basic', 'tcdf']
LABEL = {'single': 'Single network', 'basic': 'Basic template fusion', 'tcdf': 'TCDF (Ours)'}

AP_PATTERNS = {
    'AP50': r'Average Precision\s+\(AP\) @\[ IoU=0\.50\s+\| area=\s+all \| maxDets=100 \] = ([\d.-]+)',
    'AP50-95': r'Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95 \| area=\s+all \| maxDets=100 \] = ([\d.-]+)',
}


def read_metrics(log_path):
    if not log_path.is_file():
        return {}
    text = log_path.read_text(errors='ignore')
    out = {}
    for name, pattern in AP_PATTERNS.items():
        found = re.findall(pattern, text)
        if found:
            out[name] = float(found[-1])
    return out


def main(root):
    root = Path(root)
    runs = {}
    deltas = set()
    for d in sorted(root.glob('*_d*')):
        m = re.match(r'(.+)_d(\d+)$', d.name)
        if not m:
            continue
        method, delta = m.group(1), int(m.group(2))
        metrics = read_metrics(d / 'eval.log')
        if metrics:
            runs[(method, delta)] = metrics
            deltas.add(delta)

    if not runs:
        print(f'no completed runs under {root}')
        return

    deltas = sorted(deltas)
    for metric in ('AP50', 'AP50-95'):
        print(f'\n=== {root.name}  {metric} ===')
        header = f'{"method":<24}' + ''.join(f'{"d=" + str(d):>9}' for d in deltas) + f'{"drop@max":>10}'
        print(header)
        print('-' * len(header))
        for method in METHODS:
            row = f'{LABEL[method]:<24}'
            vals = []
            for d in deltas:
                v = runs.get((method, d), {}).get(metric)
                vals.append(v)
                row += f'{v:>9.4f}' if v is not None else f'{"-":>9}'
            if vals and vals[0] is not None and vals[-1] is not None:
                row += f'{vals[0] - vals[-1]:>+10.4f}'
            print(row)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'output/align_phase1/deeppcb')
