"""Export a complete lattice at a requested IP spot-size pair."""
import argparse
from . import write_lattice_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sigma-x-mm', type=float, required=True)
parser.add_argument('--sigma-y-mm', type=float, required=True)
parser.add_argument('--output', required=True, help='Cheetah-format lattice JSON')
args = parser.parse_args()
tune = write_lattice_json(args.output, args.sigma_x_mm * 1e-3, args.sigma_y_mm * 1e-3)
print(f"Saved {len(tune['settings'])} quadrupole strengths to {args.output}")
