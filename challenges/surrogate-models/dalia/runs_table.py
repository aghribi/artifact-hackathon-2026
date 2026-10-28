"""One line per training run under outputs/runs/ (their result.json), newest first."""

import json
import sys
from pathlib import Path

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/runs")
results = [json.loads(p.read_text()) for p in folder.glob("*/result.json")]
results.sort(key=lambda r: r["run_id"], reverse=True)
print(f"{'run':28s} {'model':38s} {'split':12s} {'mean R2':>8s} {'in range':>9s}  submission")
for r in results:
    m = r["metrics"]
    in_range = m.get("holdout_in_range_mean_r2")
    name = r["model_name"] + (f" v{r['registry_version']}" if r.get("registry_version") else "")
    print(f"{r['run_id']:28s} {name:38s} {r['split']:12s} {m['holdout_mean_r2']:8.4f} "
          f"{'' if in_range is None else f'{in_range:9.4f}':>9s}  {r.get('submission') or ''}")
