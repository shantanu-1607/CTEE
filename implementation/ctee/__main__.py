import argparse
from pathlib import Path
from threadpoolctl import threadpool_limits
from .runtime import DEFAULT_RUN,DEFAULT_CONFIG,settings


def main():
    p=argparse.ArgumentParser(description='CTEE: manual NumPy electrical classification pipeline')
    p.add_argument('phase',choices=['audit','generate','train','evaluate','confirm','all'])
    p.add_argument('--root',type=Path,default=DEFAULT_RUN)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--configurations',type=int,default=None)
    p.add_argument('--epochs',type=int,default=None)
    p.add_argument('--seed',type=int,default=None)
    a=p.parse_args();cfg=settings(a.config);a.root.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        if a.phase in ['audit','all']:
            from .audit import run;run(a.root)
        if a.phase in ['generate','all']:
            from .data import generate
            generate(a.root,a.configurations or cfg['configurations_per_topology'],a.seed if a.seed is not None else cfg['data_seed'])
        if a.phase in ['train','all']:
            from .train import fit;fit(a.root,a.epochs,a.config)
        if a.phase in ['evaluate','all']:
            from .evaluate import evaluate;evaluate(a.root)
        if a.phase in ['confirm','all']:
            from .confirm import confirm;confirm(a.root,seed=cfg['data_seed']+1)

if __name__=='__main__':main()
