"""Export ranked orbit SVD controls for one requested target-size pair."""
import argparse
import json
from pathlib import Path
from . import OrbitModes

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sigma-x-mm',required=True,type=float)
parser.add_argument('--sigma-y-mm',required=True,type=float)
parser.add_argument('--top',type=int,default=5)
parser.add_argument('--plane',choices=['x','y','both'],default='both')
parser.add_argument('--output',type=Path)
parser.add_argument('--backend',choices=['xsuite','cheetah'],default='xsuite')
args=parser.parse_args()
planes='xy' if args.plane=='both' else args.plane
result={p:OrbitModes(p,backend=args.backend).ranked_variables(args.sigma_x_mm*1e-3,args.sigma_y_mm*1e-3,args.top) for p in planes}
encoded=json.dumps(result,indent=2)+'\n'
if args.output:args.output.write_text(encoded)
else:print(encoded,end='')
