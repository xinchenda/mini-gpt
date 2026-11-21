"""Score complete packed validation/test streams with explicit sliding stride.

Reuses scoring_windows; follows the fixed-context likelihood approach documented
at https://huggingface.co/docs/transformers/en/perplexity .
"""
import argparse
from itertools import islice
import json
import math
from pathlib import Path
import numpy as np
import torch
from .data import digest
from .inference import load_artifact, scoring_windows


@torch.inference_mode()
def score_batched(model, ids, batch_size=8, stride=256):
    windows=iter(scoring_windows(ids,model.config.context,stride))
    total=count=0
    device=next(model.parameters()).device
    while batch:=list(islice(windows,batch_size)):
        groups={}
        for window,suffix in batch:
            groups.setdefault((len(window),suffix),[]).append(window)
        for (_,suffix),group in groups.items():
            values=torch.from_numpy(np.array(group,dtype=np.int64)).to(device)
            logits=model(values[:,:-1])[:,-suffix:].float()
            total+=torch.nn.functional.cross_entropy(logits.flatten(0,1),
                      values[:,-suffix:].flatten(),reduction='sum').item()
            count+=len(group)*suffix
    if count!=len(ids)-1:raise ValueError('Target coverage mismatch')
    return dict(targets=count,loss=total/count,perplexity=math.exp(total/count),
                stride=stride,context=model.config.context,precision='fp32',
                protocol='strided_all_targets_once')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact')
    parser.add_argument('--data',type=Path,default=Path('data/web_cloud'))
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--stride',type=int,default=256)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.batch_size<1:parser.error('batch-size must be positive')
    model,tokenizer=load_artifact(args.artifact,args.device)
    if digest(args.data/'tokenizer.json')!=digest(Path(args.artifact)/'tokenizer.json'):
        raise ValueError('Data tokenizer differs from model tokenizer')
    report=dict(device=args.device,documents={})
    for role in ('validation','test'):
        path=args.data/f'{role}.bin'
        ids=np.memmap(path,dtype='<u2',mode='r')
        result=score_batched(model,ids,args.batch_size,args.stride)
        result['sha256']=digest(path)
        report['documents'][role]=result
        print(role,result,flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
