"""Write one cycle grid per Parquet group: HBL, MBL, SPOKES, BCM."""
import collections, csv, functools, json, math, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import plot_all_cycles as plots

OUT=Path(__file__).resolve().parent
plots.font=functools.lru_cache(maxsize=16)(plots.font)
rows=json.loads((OUT/'waveforms.json').read_text())
index=json.loads((OUT/'plot_index.json').read_text())
colors={pv:tuple(bytes.fromhex(color[1:])) for pv,color in index['pvs'].items()}
groups=collections.defaultdict(lambda:collections.defaultdict(list))
for cycle,records in rows.items():
    for i,r in enumerate(records):groups[Path(r['member']).stem][cycle].append(i)

columns=8;size=(600,340);titleheight=110
summary=[];started=time.time()
for group in ['HBL','MBL','SPOKES','BCM']:
    cycles=sorted(groups[group],key=int)
    selected=[rows[c][i] for c in cycles for i in groups[group][c]]
    pvs=sorted(set(r['pv'] for r in selected))
    nonempty=sum(r['min'] is not None for r in selected)
    canvas=Image.new('RGB',(columns*size[0],math.ceil(len(cycles)/columns)*size[1]+titleheight),'#e8edf3')
    draw=ImageDraw.Draw(canvas)
    draw.text((25,18),f'{group}: all PV waveforms, separated by cycle_id',font=plots.font(32),fill='#152536')
    draw.text((25,61),f'{len(cycles)} cycles | {len(pvs)} PVs | {len(selected):,} rows | {nonempty:,} nonempty waveforms | dd.list (duplicates included)',font=plots.font(18),fill='#435365')
    draw.text((25,86),'Raw amplitude; sample-index x axis; independent y scale per cycle; 256-bin min/max envelope.',font=plots.font(15),fill='#435365')
    for j,cycle in enumerate(cycles):
        ids=groups[group][cycle];records=[rows[cycle][i] for i in ids]
        data=np.memmap(OUT/'cache'/f'{cycle}.f32',dtype='<f4',mode='r',shape=(len(rows[cycle]),plots.BINS*2))
        panel=plots.render_panel(cycle,records,data[ids],colors,size)
        canvas.paste(panel,((j%columns)*size[0],(j//columns)*size[1]+titleheight))
        if (j+1)%100==0:print(f'{group}: {j+1}/{len(cycles)} panels ({time.time()-started:.0f}s)',flush=True)
        del data,panel
    filename=f'all_cycles_{group}.png'
    width,height=canvas.size
    canvas.save(OUT/filename,compress_level=3)
    canvas.thumbnail((2400,16000));canvas.save(OUT/f'all_cycles_{group}_preview.jpg',quality=85)
    summary.append({'group':group,'file':filename,'cycles':len(cycles),'pvs':len(pvs),'rows':len(selected),
                    'nonempty_waveforms':nonempty,'width':width,'height':height})
    del draw,canvas
    print(f'Saved {filename} ({time.time()-started:.0f}s)',flush=True)
assert sum(s['rows'] for s in summary)==sum(map(len,rows.values()))
assert sum(s['nonempty_waveforms'] for s in summary)==218527
(OUT/'group_overviews.json').write_text(json.dumps(summary,indent=2))
with (OUT/'group_overviews.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
print('All group overviews complete.',flush=True)
