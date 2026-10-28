"""A live status page: the best model of each case in the Hopsworks registry next to its
official score on the W&B leaderboard.

Run from this folder with the environment's own Python:
    ../../../.venv/bin/python status.py              # writes outputs/status.html and opens it
    ../../../.venv/bin/python status.py --watch 300  # rebuilds every 5 minutes; the page reloads

status_app.py serves the same page as a Streamlit app in Hopsworks (Apps), refreshed live.
Both sources are read only. The registry scores are each author's own holdout (not comparable
across authors); `pallas/mean_r2` on W&B is the organisers' score on the hidden test, the one
the leaderboard ranks on. Official runs name the registry model they scored in their config
(`model_name`, `model_version`), which is how the two are matched.
"""

import argparse
import html
import json
import os
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import requests

import pallas_pipeline as pp
from team_wandb import ENTITY, GROUP, PROJECT as WANDB_PROJECT, load_env

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "status.html"
TIER_ORDER = {"easy": 1, "medium": 2, "hard": 3}

# (task as logged on W&B, title, tier, bar on the mean R2 or None, bar as objectives.md says it)
OBJECTIVES = [
    ("inverse_A", "Inverse A", "easy", 0.9100, "reference ensemble, mean R² 0.9100"),
    ("direct_B", "Direct B", "easy", (0.94 + 0.26 + 0.81) / 3,
     "kickoff linear baseline, about 0.94 / 0.26 / 0.81"),
    ("trajectory", "Trajectory", "medium", 0.8960,
     "internal emulator in plasma, mean R² 0.8960 (Easy: beat your own linear model)"),
    ("inverse_B", "Inverse B", "medium", 0.6480, "linear baseline, mean R² 0.6480"),
    ("moments", "Moments at the planes", "medium", None, "R² per column, no bar published"),
    ("ood", "Out of distribution in density", "hard", None,
     "no bar: the deliverable is the in-range vs above-cut gap, honestly measured"),
]
CASE_OF_TASK = {"inverse_A": "inverse_A", "direct_B": "direct_B", "trajectory": "trajectory",
                "inverse_B": "inverse_B", "ood": "trajectory"}


# ---- sources ----------------------------------------------------------------------------------

def task_of_model(name):
    """The W&B task a registry model belongs to, from the case its name carries."""
    name = name.lower()
    if "ood" in name:
        return "ood"
    for task in ("trajectory", "direct_B", "inverse_B", "inverse_A", "moments"):
        if task.lower() in name:
            return task
    return None


def targets_of(task):
    return pp.CASES[CASE_OF_TASK[task]].targets if task in CASE_OF_TASK else []


def registry_entries():
    """The raw model-registry listing of the team project.

    With HOPSWORKS_API_KEY (a laptop): the REST API. Without it (inside Hopsworks, e.g. the app):
    the session's own client, which needs no key.
    """
    if not os.environ.get("HOPSWORKS_API_KEY"):
        import hopsworks
        from hopsworks_common import client
        project = hopsworks.login()
        path = ["project", project.id, "modelregistries", project.id, "models"]
        return client._get_instance()._send_request("GET", path).get("items", [])
    host = os.environ["HOPSWORKS_HOST"]
    headers = {"Authorization": f"ApiKey {os.environ['HOPSWORKS_API_KEY']}"}
    api = f"https://{host}/hopsworks-api/api/project"
    projects = requests.get(api, headers=headers, timeout=30)
    projects.raise_for_status()
    project_id = next(p.get("project", p)["id"] for p in projects.json()
                      if p.get("project", p)["name"] == pp.PROJECT)
    answer = requests.get(f"{api}/{project_id}/modelregistries/{project_id}/models",
                          headers=headers, timeout=60)
    answer.raise_for_status()
    return answer.json().get("items", [])


def registry_models():
    """Every registered model of the team project, with its holdout scores."""
    models = []
    for entry in registry_entries():
        task = task_of_model(entry["name"])
        metrics = entry.get("metrics") or {}
        per_target = {t: metrics.get(f"holdout_r2_{pp.fs_name(t)}", metrics.get(f"r2_{pp.fs_name(t)}"))
                      for t in targets_of(task)}
        mean = metrics.get("holdout_mean_r2", metrics.get("r2_mean"))
        derived = mean is None and per_target and all(v is not None for v in per_target.values())
        if derived:   # some authors log the targets only
            mean = sum(per_target.values()) / len(per_target)
        models.append({
            "name": entry["name"], "version": entry["version"], "task": task,
            "author": entry.get("userFullName", ""),
            "created": datetime.fromtimestamp(entry["created"] / 1000),
            "mean": mean, "derived": derived, "per_target": per_target,
            "in_range": metrics.get("holdout_in_range_mean_r2"), "gap": metrics.get("holdout_gap"),
            "description": entry.get("description") or ""})
    return models


def wandb_key():
    """WANDB_API_KEY from the environment / .env, else the Hopsworks secret of that name."""
    if not os.environ.get("WANDB_API_KEY"):
        import hopsworks
        hopsworks.login()
        os.environ["WANDB_API_KEY"] = hopsworks.get_secrets_api().get_secret("WANDB_API_KEY").value
    return os.environ["WANDB_API_KEY"]


RUNS_QUERY = """query Runs($entity: String!, $project: String!, $cursor: String) {
  project(name: $project, entityName: $entity) {
    runs(first: 200, after: $cursor) {
      edges { node { name displayName group state createdAt config summaryMetrics } }
      pageInfo { hasNextPage endCursor } } } }"""


def wandb_runs_graphql():
    """The same as wandb_runs, through W&B's public GraphQL API (no wandb package needed)."""
    runs, cursor = [], None
    while True:
        answer = requests.post("https://api.wandb.ai/graphql", auth=("api", wandb_key()),
                               json={"query": RUNS_QUERY, "variables": {
                                   "entity": ENTITY, "project": WANDB_PROJECT, "cursor": cursor}},
                               timeout=60)
        answer.raise_for_status()
        listing = answer.json()["data"]["project"]["runs"]
        for edge in listing["edges"]:
            node = edge["node"]
            config = {k: v.get("value") if isinstance(v, dict) else v
                      for k, v in json.loads(node.get("config") or "{}").items()}
            summary = {k: v for k, v in json.loads(node.get("summaryMetrics") or "{}").items()
                       if not k.startswith("_") and isinstance(v, (int, float))}
            runs.append({
                "name": node["displayName"],
                "url": f"https://wandb.ai/{ENTITY}/{WANDB_PROJECT}/runs/{node['name']}",
                "group": node.get("group") or "(no group)", "state": node["state"],
                "task": config.get("task"), "tier": config.get("tier"),
                "created": node["createdAt"], "scored_at": config.get("scored_at", ""),
                "model": config.get("model_name"), "model_version": config.get("model_version"),
                "official": "pallas/mean_r2" in summary, "summary": summary})
        if not listing["pageInfo"]["hasNextPage"]:
            return runs
        cursor = listing["pageInfo"]["endCursor"]


def wandb_runs():
    """Every run of the challenge project the team key can see, with its summary."""
    try:
        import wandb
    except ImportError:            # e.g. Hopsworks' app environment
        return wandb_runs_graphql()
    api = wandb.Api(api_key=wandb_key(), timeout=60)
    runs = []
    for run in api.runs(f"{ENTITY}/{WANDB_PROJECT}"):
        summary = {k: v for k, v in run.summary._json_dict.items()
                   if not k.startswith("_") and isinstance(v, (int, float))}
        config = dict(run.config)
        runs.append({
            "name": run.name, "url": run.url, "group": run.group or "(no group)",
            "state": run.state, "task": config.get("task"), "tier": config.get("tier"),
            "created": run.created_at, "scored_at": config.get("scored_at", ""),
            "model": config.get("model_name"), "model_version": config.get("model_version"),
            "official": "pallas/mean_r2" in summary, "summary": summary})
    return runs


# ---- status -----------------------------------------------------------------------------------

def best(items, key):
    scored = [i for i in items if key(i) is not None]
    return max(scored, key=key) if scored else None


def reference(task):
    """Per-target reference of objectives.md (pack_reference.json), when the pack is local."""
    if task not in ("inverse_A", "direct_B", "trajectory", "inverse_B"):
        return {}, ""
    try:
        case = pp.CASES[task]
        return pp.reference_scores(case), case.reference_note
    except (OSError, KeyError):
        return {}, ""


def case_status(objective, models, runs):
    task, title, tier, bar, bar_note = objective
    mine = [r for r in runs if r["group"] == GROUP and r["task"] == task and r["official"]]
    official = best(mine, lambda r: r["summary"]["pallas/mean_r2"])
    case_models = sorted((m for m in models if m["task"] == task),
                         key=lambda m: (m["mean"] is None, -(m["mean"] or 0)))
    submitted = {(r["model"], r["model_version"]) for r in mine}
    for m in case_models:
        m["submitted"] = (m["name"], m["version"]) in submitted
    score = official["summary"]["pallas/mean_r2"] if official else None
    if official is None:
        verdict = "none"
    elif bar is None:
        verdict = "scored"
    else:
        verdict = "beaten" if score > bar else "below"
    return {"task": task, "title": title, "tier": tier, "bar": bar, "bar_note": bar_note,
            "official": official, "all_official": sorted(mine, key=lambda r: r["created"]),
            "score": score, "verdict": verdict, "models": case_models,
            "best_model": case_models[0] if case_models and case_models[0]["mean"] is not None
            else None,
            "submitted_model": next((m for m in case_models if m["submitted"]), None)}


def standing(cases):
    """GRADING.md: highest tier with a bar beaten (a scored Hard task counts, it has no bar)."""
    reached = [c["tier"] for c in cases if c["verdict"] == "beaten"
               or (c["verdict"] == "scored" and c["tier"] == "hard")]
    return max(reached, key=TIER_ORDER.get) if reached else None


# ---- page -------------------------------------------------------------------------------------

def f4(value):
    return "—" if value is None else f"{value:.4f}"


def signed(value):
    return "" if value is None else f"{value:+.4f}"


def esc(value):
    return html.escape(str(value))


def card(c):
    official, top, sent = c["official"], c["best_model"], c["submitted_model"]
    pill = {"beaten": ("ok", "Bar beaten"), "below": ("bad", "Below the bar"),
            "scored": ("info", "Scored"), "none": ("muted", "Not submitted")}[c["verdict"]]
    parts = [f'<section class="card"><header><div><span class="tier {c["tier"]}">'
             f'{c["tier"]}</span><h2>{esc(c["title"])}</h2></div>'
             f'<span class="pill {pill[0]}">{pill[1]}</span></header>']

    # headline numbers
    parts.append('<div class="stats">')
    parts.append(f'<div class="stat"><span class="label">Official mean R² (W&amp;B)</span>'
                 f'<span class="big">{f4(c["score"])}</span>')
    if official:
        parts.append(f'<a href="{esc(official["url"])}">{esc(official["model"] or official["name"])}'
                     f'{" v" + esc(official["model_version"]) if official["model_version"] else ""}'
                     f'</a><span class="sub">scored {esc(official["scored_at"])}</span>')
    parts.append('</div>')
    if c["bar"] is not None:
        delta = c["score"] - c["bar"] if c["score"] is not None else None
        parts.append(f'<div class="stat"><span class="label">Bar</span>'
                     f'<span class="big">{f4(c["bar"])}</span>'
                     f'<span class="delta {"up" if (delta or 0) > 0 else "down"}">{signed(delta)}</span>'
                     f'<span class="sub">{esc(c["bar_note"])}</span></div>')
    elif c["task"] == "ood" and official:
        s = official["summary"]
        parts.append(f'<div class="stat"><span class="label">In range / gap</span>'
                     f'<span class="big">{f4(s.get("pallas/in_distribution/mean_r2"))}</span>'
                     f'<span class="delta down">gap {f4(s.get("pallas/drop/mean"))}</span>'
                     f'<span class="sub">{esc(c["bar_note"])}</span></div>')
    else:
        parts.append(f'<div class="stat"><span class="label">Bar</span><span class="big">—</span>'
                     f'<span class="sub">{esc(c["bar_note"])}</span></div>')
    parts.append(f'<div class="stat"><span class="label">Best holdout in the registry</span>'
                 f'<span class="big">{f4(top["mean"]) if top else "—"}</span>')
    if top:
        parts.append(f'<span>{esc(top["name"])} v{top["version"]}</span>'
                     f'<span class="sub">{esc(top["author"])}{" · submitted" if top["submitted"] else ""}'
                     f'</span>')
    parts.append('</div></div>')

    # a better holdout than the submitted model is a candidate for the next CSV
    if top and not top["submitted"] and (sent is None or (sent["mean"] or 0) < top["mean"]):
        against = f' (submitted {esc(sent["name"])} v{sent["version"]}: {f4(sent["mean"])})' if sent else ""
        parts.append(f'<p class="hint">Candidate: <b>{esc(top["name"])} v{top["version"]}</b> has the '
                     f'best holdout, {f4(top["mean"])}{against}. Holdouts are each author\'s own '
                     f'split, so check it on the same split before resubmitting.</p>')

    # per target
    targets = targets_of(c["task"])
    if targets and (official or top):
        ref, ref_note = reference(c["task"])
        s = official["summary"] if official else {}
        parts.append('<div class="table-wrap"><table class="targets"><thead><tr><th>Target</th><th>Official</th>'
                     '<th>Submitted, holdout</th><th>Best, holdout</th>'
                     f'{"<th>Reference</th>" if ref else ""}</tr></thead><tbody>')
        for t in targets:
            o = s.get(f"pallas/r2/{t}")
            r = ref.get(t)
            cls = "" if o is None or r is None else (' class="up"' if o > r else ' class="down"')
            parts.append(f'<tr><td>{esc(t)}</td><td{cls}>{f4(o)}</td>'
                         f'<td>{f4(sent["per_target"].get(t)) if sent else "—"}</td>'
                         f'<td>{f4(top["per_target"].get(t)) if top else "—"}</td>'
                         f'{f"<td>{f4(r)}</td>" if ref else ""}</tr>')
        parts.append('</tbody></table></div>')
        if ref:
            parts.append(f'<p class="note">Reference: {esc(ref_note)}.</p>')
        if c["task"] == "trajectory" and "pallas/drift/mean_r2" in s:
            parts.append(f'<p class="note">Drift after the plasma (reported, not ranked): '
                         f'{f4(s["pallas/drift/mean_r2"])}.</p>')

    # every registered model of the case
    if c["models"]:
        parts.append(f'<details><summary>{len(c["models"])} registered model(s)</summary>'
                     '<table><thead><tr><th>Model</th><th>v</th><th>Author</th>'
                     '<th>Holdout mean R²</th><th>Created</th><th></th></tr></thead><tbody>')
        for m in c["models"]:
            extra = f' <span class="sub">gap {f4(m["gap"])}</span>' if m["gap"] is not None else ""
            parts.append(f'<tr{" class=sent" if m["submitted"] else ""}><td>{esc(m["name"])}</td>'
                         f'<td>{m["version"]}</td><td>{esc(m["author"])}</td>'
                         f'<td>{f4(m["mean"])}{"*" if m["derived"] else ""}{extra}</td>'
                         f'<td>{m["created"]:%d %b %H:%M}</td>'
                         f'<td>{"submitted" if m["submitted"] else ""}</td></tr>')
        parts.append('</tbody></table><p class="note">* mean of the per-target R² '
                     '(the model logs no mean).</p></details>')
    else:
        parts.append('<p class="empty">No model in the registry yet.</p>')
    parts.append('</section>')
    return "".join(parts)


def leaderboard(runs):
    """Best official mean R2 per team and task, as far as the team key can see."""
    tasks = [o[0] for o in OBJECTIVES]
    groups = sorted({r["group"] for r in runs if r["official"]})
    if not groups:
        return '<p class="empty">No official score on W&amp;B yet.</p>'
    rows = ['<table><thead><tr><th>Team</th>' + "".join(f"<th>{esc(t)}</th>" for t in tasks)
            + '</tr></thead><tbody>']
    for g in groups:
        cells = []
        for t in tasks:
            top = best([r for r in runs if r["group"] == g and r["task"] == t and r["official"]],
                       lambda r: r["summary"]["pallas/mean_r2"])
            cells.append(f'<td><a href="{esc(top["url"])}">{f4(top["summary"]["pallas/mean_r2"])}</a></td>'
                         if top else "<td>—</td>")
        rows.append(f'<tr{" class=sent" if g == GROUP else ""}><td>{esc(g)}</td>{"".join(cells)}</tr>')
    rows.append('</tbody></table>')
    if groups == [GROUP]:
        rows.append('<p class="note">Only our group is visible to the team key: other teams\' '
                    'runs are private to them, so their scores show on Agora-Conf, not here.</p>')
    return "".join(rows)


def runs_table(runs):
    rows = ['<table><thead><tr><th>Run</th><th>Group</th><th>Task</th><th>Tier</th>'
            '<th>Official mean R²</th><th>Holdout mean R²</th><th>Created</th></tr></thead><tbody>']
    for r in sorted(runs, key=lambda r: r["created"], reverse=True):
        rows.append(f'<tr><td><a href="{esc(r["url"])}">{esc(r["name"])}</a></td>'
                    f'<td>{esc(r["group"])}</td><td>{esc(r["task"] or "")}</td>'
                    f'<td>{esc(r["tier"] or "")}</td>'
                    f'<td>{f4(r["summary"].get("pallas/mean_r2"))}</td>'
                    f'<td>{f4(r["summary"].get("holdout/mean_r2"))}</td>'
                    f'<td>{esc(r["created"][:16].replace("T", " "))}</td></tr>')
    rows.append('</tbody></table>')
    return "".join(rows)


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#16181d;--muted:#667085;--line:#e4e7ec;--accent:#3b5bdb;
--ok:#1a7f4b;--ok-bg:#e3f5ea;--bad:#b42318;--bad-bg:#fdeceb;--info:#3b5bdb;--info-bg:#e8edff;
--easy:#1a7f4b;--medium:#b54708;--hard:#b42318;--hint:#fff8e6;--sent:#f0f4ff}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#171a21;--ink:#e6e8ec;--muted:#98a2b3;
--line:#2a2f3a;--accent:#8ea2ff;--ok:#5fd394;--ok-bg:#12301f;--bad:#ff8a80;--bad-bg:#3a1714;
--info:#8ea2ff;--info-bg:#1c2340;--easy:#5fd394;--medium:#f5b267;--hard:#ff8a80;--hint:#2e2814;
--sent:#1c2236}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,
"Segoe UI",Inter,Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 16px 48px}
h1{font-size:22px;margin:0}h2{font-size:16px;margin:0;display:inline}h3{font-size:15px;margin:32px 0 10px}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.top{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;flex-wrap:wrap;
margin-bottom:18px}.meta{color:var(--muted);font-size:13px}
.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(180px,100%),1fr));gap:12px;margin-bottom:20px}
.summary div{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.summary b{display:block;font-size:20px;margin-top:2px;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(520px,100%),1fr));gap:16px}
@media (max-width:600px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;min-width:0}
.card header{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:12px}
.tier{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;margin-right:8px}
.tier.easy{color:var(--easy)}.tier.medium{color:var(--medium)}.tier.hard{color:var(--hard)}
.pill{font-size:12px;font-weight:600;padding:3px 10px;border-radius:999px;white-space:nowrap}
.pill.ok{color:var(--ok);background:var(--ok-bg)}.pill.bad{color:var(--bad);background:var(--bad-bg)}
.pill.info{color:var(--info);background:var(--info-bg)}.pill.muted{color:var(--muted);background:var(--bg)}
.stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
@media (max-width:600px){.stats{grid-template-columns:1fr}}
.stat{display:flex;flex-direction:column;gap:2px;min-width:0;overflow-wrap:anywhere}
.label{font-size:12px;color:var(--muted)}.big{font-size:24px;font-weight:600;font-variant-numeric:tabular-nums}
.sub,.note{font-size:12px;color:var(--muted)}.delta{font-size:13px;font-weight:600}
.up{color:var(--ok)}.down{color:var(--bad)}
.hint{background:var(--hint);border-radius:8px;padding:8px 10px;margin:12px 0 0;font-size:13px}
.empty{color:var(--muted);margin:12px 0 0}
.table-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;margin-top:12px;font-variant-numeric:tabular-nums;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-size:12px;color:var(--muted);font-weight:500}
tr.sent td{background:var(--sent)}
details{margin-top:12px}summary{cursor:pointer;color:var(--muted);font-size:13px}
details table{display:block;overflow-x:auto}
.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:4px 18px 14px;
overflow-x:auto}
.error{background:var(--bad-bg);color:var(--bad);border-radius:8px;padding:10px 12px;margin-bottom:12px}
"""


def page(cases, runs, models, errors, refresh, live=None):
    now = datetime.now()
    reached = standing(cases)
    official = sum(1 for c in cases if c["official"])
    summary = (
        f'<div><span class="label">Standing (GRADING.md)</span><b>{esc((reached or "—").title())}</b></div>'
        f'<div><span class="label">Tasks with an official score</span><b>{official} / {len(cases)}</b></div>'
        f'<div><span class="label">Bars beaten</span>'
        f'<b>{sum(c["verdict"] == "beaten" for c in cases)} / {sum(c["bar"] is not None for c in cases)}</b></div>'
        f'<div><span class="label">Models in the registry</span><b>{len(models)}</b></div>'
        f'<div><span class="label">W&amp;B runs visible</span><b>{len(runs)}</b></div>')
    meta_refresh = f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""
    every = f" · rebuilt every {refresh}s" if refresh else ""
    if live:
        every = f" · live, refreshed every {live}s"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
{meta_refresh}<title>Plasma Pirates Status</title><style>{CSS}</style></head>
<body><main>
<div class="top"><div><h1>Plasma Pirates · surrogate models</h1>
<div class="meta">Hopsworks <b>{esc(pp.PROJECT)}</b> registry · W&amp;B <b>{esc(ENTITY)}/{esc(WANDB_PROJECT)}</b>,
group <b>{esc(GROUP)}</b></div></div>
<div class="meta">Updated {now:%a %d %b %Y, %H:%M:%S}{every}</div></div>
{"".join(f'<div class="error">{esc(e)}</div>' for e in errors)}
<div class="summary">{summary}</div>
<div class="grid">{"".join(card(c) for c in cases)}</div>
<h3>Leaderboard (best official mean R² per team)</h3>
<div class="panel">{leaderboard(runs)}</div>
<h3>W&amp;B runs</h3>
<div class="panel">{runs_table(runs) if runs else '<p class="empty">No runs.</p>'}</div>
<p class="note" style="margin-top:20px">Official = <code>pallas/mean_r2</code>, the organisers' score on
the hidden test. Holdout = the score each model was registered with, on its author's own split:
a guide for choosing what to submit, not a ranking.</p>
</main></body></html>"""


def collect():
    """(cases, runs, models, errors): everything the page shows, read once."""
    load_env()
    errors = []
    try:
        models = registry_models()
    except Exception as error:
        models = []
        errors.append(f"Hopsworks registry: {type(error).__name__}: {str(error)[:200]}")
    try:
        runs = wandb_runs()
    except Exception as error:
        runs = []
        errors.append(f"W&B: {type(error).__name__}: {str(error)[:200]}")
    cases = [case_status(o, models, runs) for o in OBJECTIVES]
    return cases, runs, models, errors


def render(refresh=None, live=None):
    """(the page as an HTML string, errors); `live` labels it as refreshed every `live` s."""
    cases, runs, models, errors = collect()
    return page(cases, runs, models, errors, refresh, live), errors, cases


def build(out, refresh):
    html_page, errors, cases = render(refresh)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_page)
    for c in cases:
        print(f"  {c['task']:<11} official {f4(c['score'])}  bar {f4(c['bar'])}  "
              f"best holdout {f4(c['best_model']['mean']) if c['best_model'] else '—'}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT, help=f"HTML file (default {OUT})")
    parser.add_argument("--watch", type=int, metavar="SECONDS",
                        help="rebuild every SECONDS; the page reloads itself at the same pace")
    parser.add_argument("--no-open", action="store_true", help="don't open the page in a browser")
    args = parser.parse_args()
    opened = args.no_open
    while True:
        print(f"{datetime.now():%H:%M:%S} reading the registry and W&B ...")
        for error in build(args.out, args.watch):
            print(f"  ! {error}")
        print(f"  wrote {args.out}")
        if not opened:
            webbrowser.open(args.out.resolve().as_uri())
            opened = True
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
