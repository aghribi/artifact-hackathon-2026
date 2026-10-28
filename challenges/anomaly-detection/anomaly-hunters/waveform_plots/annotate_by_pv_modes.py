"""Add beam_mode to existing by_pv PNG headers, preserving plotted pixels.

Run with: python waveform_plots/annotate_by_pv_modes.py
The regular plot_by_pv.py generator includes the same labels on future runs.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import stat
import time

from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo

from beam_modes import load_beam_modes, mode_caption, panel_mode
from plot_all_cycles import font


def annotate(pv=None):
    root = Path(__file__).resolve().parent
    output = root / "by_pv"
    waveforms = json.loads((root / "waveforms.json").read_text())
    modes = load_beam_modes(root, waveforms)
    mode_digest = json.loads((root / "waveform_beam_modes.json").read_text())["inventory_sha256"]
    manifest = json.loads((output / "manifest.json").read_text())
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for cycle, records in waveforms.items():
        for index, record in enumerate(records):
            grouped[record["pv"]][cycle].append(index)
    selected = [r for r in manifest if pv is None or r["pv"] == pv]
    if not selected:
        raise ValueError(f"PV not found: {pv}")
    label_font = font(12)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    # Check every caption fits before changing any images.
    for record in selected:
        for cycle, indices in grouped[record["pv"]].items():
            mode = panel_mode(modes, cycle, indices)
            nonempty = sum(waveforms[cycle][i]["min"] is not None for i in indices)
            if probe.textlength(mode_caption(mode, nonempty, len(indices)), font=label_font) > 340:
                raise ValueError(f"Mode caption too wide: {record['pv']} / {cycle} / {mode}")

    started = time.monotonic()
    for number, record in enumerate(selected, 1):
        path = output / record["file"]
        cycles = sorted(grouped[record["pv"]], key=int)
        columns = min(6, len(cycles))
        expected_size = (columns * 360, ((len(cycles) + columns - 1) // columns) * 220 + 102)
        with Image.open(path) as image:
            if image.size != expected_size or image.size != (record["width"], record["height"]):
                raise ValueError(f"Unexpected plot layout: {path}")
            if (image.info.get("beam_mode_annotation") == "1"
                    and image.info.get("beam_mode_source") == mode_digest):
                continue
            image.load()
            draw = ImageDraw.Draw(image)
            for position, cycle in enumerate(cycles):
                x = (position % columns) * 360
                y = (position // columns) * 220 + 102
                indices = grouped[record["pv"]][cycle]
                nonempty = sum(waveforms[cycle][i]["min"] is not None for i in indices)
                mode = panel_mode(modes, cycle, indices)
                draw.rectangle((x, y + 23, x + 359, y + 41), fill="white")
                draw.text((x + 10, y + 24), mode_caption(mode, nonempty, len(indices)),
                          font=label_font, fill="#175c47")
            metadata = PngInfo()
            for key, value in image.info.items():
                if isinstance(value, str):
                    metadata.add_text(key, value)
            metadata.add_text("beam_mode_annotation", "1")
            metadata.add_text("beam_mode_source", mode_digest)
            metadata.add_text("beam_mode_panels", str(len(cycles)))
            partial = path.with_suffix(".png.partial")
            image.save(partial, format="PNG", compress_level=1, pnginfo=metadata)
            os.chmod(partial, stat.S_IMODE(path.stat().st_mode))
            os.replace(partial, path)
        if number % 20 == 0 or number == len(selected):
            print(f"Modes added: {number}/{len(selected)} PNGs ({time.monotonic() - started:.0f}s)", flush=True)

    if pv is None:
        index_path = output / "index.html"
        page = index_path.read_text()
        page = page.replace("cycle_idごとにパネルを分けています。",
                            "cycle_idごとにパネルを分け、運転モード（beam_mode）を表示しています。")
        index_path.write_text(page)
    print(f"Complete: {len(selected)} PV plots with capture-level beam modes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pv", help="Update one PV for a preview before updating all plots")
    annotate(**vars(parser.parse_args()))
