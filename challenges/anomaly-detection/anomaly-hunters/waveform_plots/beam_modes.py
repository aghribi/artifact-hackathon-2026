"""Read capture-level beam modes without loading the full waveform pickle."""
import collections
import hashlib
import json
import os
from pathlib import Path
import tarfile

from plot_all_cycles import read_column


def load_beam_modes(root, waveforms):
    """Return mode lists in exactly the same order as waveforms.json records."""
    root = Path(root)
    inventory_bytes = (root / "inventory.json").read_bytes()
    digest = hashlib.sha256(inventory_bytes).hexdigest()
    cache_path = root / "waveform_beam_modes.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("inventory_sha256") == digest:
            modes = cached["modes"]
            _validate_modes(modes, waveforms)
            return modes

    inventory = json.loads(inventory_bytes)
    archives = collections.defaultdict(list)
    for source in inventory["files"]:
        archives[source["archive"]].append(source)
    modes = {cycle: [] for cycle in waveforms}
    for archive_index, (path, sources) in enumerate(archives.items(), 1):
        with tarfile.open(path, "r:*") as archive:
            for source in sources:
                position = 0
                with archive.extractfile(source["member"]) as stream:
                    for group in source["metadata"]["4"]:
                        columns = {c["3"]["3"][0]: {int(k): v for k, v in c["3"].items()}
                                   for c in group["1"]}
                        cycle_ids = read_column(stream, columns["cycle_id"])
                        pvs = read_column(stream, columns["pv"])
                        values = (read_column(stream, columns["beam_mode"])
                                  if "beam_mode" in columns else [None] * group["3"])
                        end = position + group["3"]
                        if (cycle_ids != source["cycle_ids"][position:end]
                                or pvs != source["pvs"][position:end]
                                or len(values) != group["3"]):
                            raise ValueError(f"Capture metadata changed: {path}:{source['member']}")
                        for cycle_id, pv, mode in zip(cycle_ids, pvs, values):
                            cycle = str(cycle_id)
                            record = waveforms[cycle][len(modes[cycle])]
                            if (record["pv"] != pv or record["archive"] != archive_index
                                    or record["member"] != source["member"]):
                                raise ValueError(f"Waveform/mode ordering mismatch: {cycle} / {pv}")
                            if mode is not None and not isinstance(mode, str):
                                raise TypeError(f"Invalid beam_mode: {mode!r}")
                            modes[cycle].append(mode)
                        position = end
                if position != source["rows"]:
                    raise ValueError(f"Capture count mismatch: {path}:{source['member']}")
        print(f"Beam modes: {archive_index}/{len(archives)} archives", flush=True)

    _validate_modes(modes, waveforms)
    partial = cache_path.with_suffix(".json.partial")
    partial.write_text(json.dumps({"inventory_sha256": digest, "modes": modes}))
    os.replace(partial, cache_path)
    counts = collections.Counter(mode or "Unknown" for values in modes.values() for mode in values)
    print(f"Beam modes verified for {sum(counts.values()):,} captures: {dict(counts)}", flush=True)
    return modes


def _validate_modes(modes, waveforms):
    if set(modes) != set(waveforms):
        raise ValueError("Beam mode cycles do not match waveform cycles")
    if any(len(modes[cycle]) != len(records) for cycle, records in waveforms.items()):
        raise ValueError("Beam mode counts do not match waveform captures")


def panel_mode(modes, cycle, selected_indices):
    """Keep all modes if the same PV/cycle has multiple captures."""
    return ", ".join(sorted({modes[cycle][i] or "Unknown" for i in selected_indices}))


def mode_caption(mode, nonempty, total):
    return f"beam_mode: {mode} | {nonempty}/{total} numeric"
