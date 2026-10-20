import argparse
from pathlib import Path
from threadpoolctl import threadpool_limits

def main():
    p=argparse.ArgumentParser(description='CTEE LoadLab phase-by-phase research pipeline')
    p.add_argument('phase',choices=['audit','generate','train','evaluate','confirm','all'])
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--configurations',type=int,default=100)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--seed',type=int,default=20261020)
    a=p.parse_args()
    with threadpool_limits(limits=1):
        if a.phase in ['audit','all']:
            from .audit import run;run(a.root)
        if a.phase in ['generate','all']:
            from .data import generate;generate(a.root,a.configurations,a.seed)
        if a.phase in ['train','all']:
            from .train import fit;fit(a.root,a.epochs)
        if a.phase in ['evaluate','all']:
            from .evaluate import evaluate;evaluate(a.root)
        if a.phase in ['confirm','all']:
            from .confirm import confirm;confirm(a.root)

if __name__=='__main__':main()
