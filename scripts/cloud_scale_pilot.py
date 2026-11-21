"""Measure a 98M candidate after the 46M pilot; never run both concurrently."""
import json
try:
    from .autodl_poweroff import power_off
except ImportError:
    from autodl_poweroff import power_off
from pathlib import Path
import subprocess
import time


def main():
    shared=Path('/root/autodl-fs')
    first=shared/'cloud_pilot_report.json'
    deadline=time.monotonic()+900
    while not first.exists():
        if time.monotonic()>deadline:
            raise TimeoutError('First pilot did not finish')
        time.sleep(5)
    root=Path('/root/autodl-tmp/minigpt')
    (root/'configs/v2-cloud98m.json').write_text(json.dumps(dict(context=512,width=768,layers=12,heads=12)))
    report=json.loads(first.read_text())
    command=list(report['command'])
    for flag,value in (('--config','configs/v2-cloud98m.json'),
                       ('--output','experiments/cloud/pilot98'),('--steps','122071')):
        command[command.index(flag)+1]=value
    subprocess.run(command,cwd=root,check=True,timeout=720)
    metrics=json.loads((root/'experiments/cloud/pilot98/metrics.jsonl').read_text().splitlines()[-1])
    result=dict(metrics=metrics,command=command,target_tokens=122071*16384,
                parameters=98032128,hourly_price_rmb=1.88,
                estimated_training_rmb=122071*16384/metrics['tokens_per_second']/3600*1.88,
                safety_factor=1.3)
    (shared/'cloud_pilot98_report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException:
        power_off()
        raise
