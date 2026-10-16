"""Write CLARA_cheetah_second_order.json: the challenge lattice with every Drift, Quadrupole
and Dipole tracked at second order (and second-order dipole edges) instead of linear.

Run from challenges/optimisation:  python optics_matching_vbc_on/second_order_lattice.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "data" / "clara" / "lattice_data" / "CLARA_cheetah.json"

lattice = json.loads(SOURCE.read_text())
for kind, params in lattice["elements"].values():
    if params.get("tracking_method") == "linear":
        params["tracking_method"] = "second_order"
    if params.get("fringe_type") == "linear_edge":
        params["fringe_type"] = "second_order_edge"
(HERE / "CLARA_cheetah_second_order.json").write_text(json.dumps(lattice))
