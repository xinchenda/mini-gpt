"""Select an affordable pilot, resume training, export and shut down.

Requires the platform shutdown deadline to have been extended separately.
"""
import json
try:
    from .autodl_poweroff import power_off
except ImportError:
    from autodl_poweroff import power_off
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT=Path('/root/autodl-tmp/minigpt')
SHARED=Path('/root/autodl-fs')


def main():
    reports=[json.loads((SHARED/name).read_text()) for name in
             ('cloud_pilot98_report.json','cloud_pilot_report.json')]
    chosen=None
    for report in reports:
        if report['estimated_training_rmb']*1.3 <= 27:
            chosen=report
            break
    if chosen is None:
        raise RuntimeError('Neither fixed training plan fits conservative budget; no formal run started')
    command=list(chosen['command'])
    pilot=ROOT/command[command.index('--output')+1]
    run=ROOT/'experiments/cloud/formal'
    if run.exists():
        raise FileExistsError(run)
    shutil.move(pilot,run)
    index=command.index('--stop-at')
    del command[index:index+2]
    command[command.index('--output')+1]='experiments/cloud/formal'
    command[command.index('--max-seconds')+1]='25200'
    command.extend(['--resume',str(run/'last.pt')])
    status=dict(state='training',command=command,
                predicted_training_rmb=chosen['estimated_training_rmb'],
                conservative_prediction_rmb=chosen['estimated_training_rmb']*1.3,
                planned_tokens=chosen['target_tokens'],hourly_price_rmb=1.88,
                hard_training_seconds=25200,
                note='Platform power-off deadline also applies; paid setup and evaluation are separate')
    (SHARED/'cloud_status.json').write_text(json.dumps(status,indent=2))
    subprocess.run(command,cwd=ROOT,check=True,timeout=25320)
    last=json.loads((run/'metrics.jsonl').read_text().splitlines()[-1])
    planned_steps=int(command[command.index('--steps')+1])
    status.update(state='exporting',last_step=last['step'],
                  completed_schedule=last['step']==planned_steps)
    (SHARED/'cloud_status.json').write_text(json.dumps(status,indent=2))
    artifact=ROOT/'artifacts/cloud'
    subprocess.run([sys.executable,'-m','minigpt_v2.export',str(run),str(artifact)],
                   cwd=ROOT,check=True,timeout=300)
    # Full optimizer states remain available after power-off in shared storage.
    shutil.copytree(run,SHARED/'cloud_training_state')
    with zipfile.ZipFile(SHARED/'cloud_model.zip','w',zipfile.ZIP_DEFLATED,compresslevel=1) as bundle:
        for path in artifact.iterdir():
            bundle.write(path,Path('artifacts/cloud')/path.name)
    shutil.copy2(ROOT/'data/web_cloud/audit.json',SHARED/'cloud_data_audit.json')
    subprocess.run([sys.executable,'-m','pip','freeze'],
                   stdout=(SHARED/'cloud_requirements.txt').open('w'),check=True,timeout=60)
    # Public benchmark evaluation has its own time limit and failure record.
    evaluation=[sys.executable,'-m','minigpt_v2.lambada',str(artifact),
                '--device','cuda','--output',str(SHARED/'cloud_lambada.json')]
    try:
        subprocess.run(evaluation,cwd=ROOT,check=True,timeout=1200)
        status['public_evaluation']='completed; contamination audit remains a limitation'
    except Exception as error:
        status['public_evaluation']=repr(error)
    status['state']='saved; shutdown requested'
    (SHARED/'cloud_status.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status,indent=2),flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        (SHARED/'cloud_failure.json').write_text(json.dumps(dict(error=repr(error)),indent=2))
        raise
    finally:
        power_off()
