"""Smallest subset of (screen, TDC voltage) shots at nominal optics that gives rank 21."""

import common
import numpy as np
import observability as obs

machine, model, beam = common.make_pair(particles=20000)
sigma = common.covariance(beam)
screens = model.inventory.screens
after = [
    s
    for s in screens
    if model.element_index[s] > model.element_index["cla_s07_dia_tdc_01"]
]
maps = {v: obs.maps_for(model, screens, v) for v in (0.0, 5e6, -5e6)}
shots = [(s, v) for v in (0.0, 5e6, -5e6) for s in screens if v == 0.0 or s in after]
rows = {shot: obs.scaled_system([maps[shot[1]][shot[0]]], sigma) for shot in shots}
chosen = []
current = np.zeros((0, 21))
while True:
    rank_now = (
        np.linalg.matrix_rank(
            current, tol=1e-8 * (np.linalg.norm(current, 2) if len(current) else 1)
        )
        if len(current)
        else 0
    )
    if rank_now == 21:
        break
    best = None
    for shot in shots:
        if shot in chosen:
            continue
        trial = np.vstack([current, rows[shot]])
        sv = np.linalg.svd(trial, compute_uv=False)
        r = int((sv > sv[0] * 1e-8).sum())
        score = (r, np.sum(np.log(sv[:r])))
        if best is None or score > best[0]:
            best = (score, shot)
    chosen.append(best[1])
    current = np.vstack([current, rows[best[1]]])
    print(len(chosen), best[1], "rank", best[0][0])
sv = np.linalg.svd(current, compute_uv=False)
print("n shots", len(chosen), "cond", sv[0] / sv[-1])
