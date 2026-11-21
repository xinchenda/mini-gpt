"""Wait for the prepared upload, validate it, and run a bounded CUDA pilot."""
import hashlib
import json
try:
    from .autodl_poweroff import power_off
except ImportError:
    from autodl_poweroff import power_off
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def main():
    root = Path('/root/autodl-tmp/minigpt')
    shared = Path('/root/autodl-fs')
    archive = shared / 'cloud_training_data.zip'
    deadline = time.monotonic() + 900
    while not zipfile.is_zipfile(archive):
        if time.monotonic() >= deadline:
            raise TimeoutError('Prepared data did not arrive within 15 minutes')
        time.sleep(5)
    with zipfile.ZipFile(archive) as bundle:
        expected = {f'data/web_cloud/{name}' for name in
                    ('train.bin','validation.bin','test.bin','tokenizer.json','manifest.json','audit.json')}
        if set(bundle.namelist()) != expected:
            raise ValueError('Unexpected archive contents')
        bundle.extractall(root)
    audit = json.loads((root/'data/web_cloud/audit.json').read_text())
    for name, expected_sha in audit['sha256'].items():
        sha = hashlib.sha256()
        with (root/'data/web_cloud'/name).open('rb') as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                sha.update(chunk)
        if sha.hexdigest() != expected_sha:
            raise ValueError(f'Data fingerprint mismatch: {name}')
    command = [sys.executable,'-m','minigpt_v2.train','--manifest','data/web_cloud/manifest.json',
               '--config','configs/v2-cloud46m.json','--output','experiments/cloud/pilot',
               '--steps','61036','--stop-at','100','--batch-size','32','--accumulation','1',
               '--device','cuda','--precision','bf16','--learning-rate','0.0006',
               '--min-learning-rate','0.00006','--max-seconds','600']
    subprocess.run(command, cwd=root, check=True, timeout=720)
    metrics = json.loads((root/'experiments/cloud/pilot/metrics.jsonl').read_text().splitlines()[-1])
    report = dict(metrics=metrics, command=command, target_tokens=61036*16384,
                  hourly_price_rmb=1.88,
                  estimated_training_rmb=61036*16384/metrics['tokens_per_second']/3600*1.88,
                  note='Measured pilot throughput includes validation and checkpoint writing')
    (shared/'cloud_pilot_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        power_off()
        raise
