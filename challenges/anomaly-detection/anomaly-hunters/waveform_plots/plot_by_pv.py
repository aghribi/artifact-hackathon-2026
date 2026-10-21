"""Write one PNG per PV, labeling each cycle_id panel with its beam_mode."""
import collections, csv, html, json, math, re, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from plot_all_cycles import BINS, font
from beam_modes import load_beam_modes, mode_caption, panel_mode

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'by_pv'
OUT.mkdir(exist_ok=True)
rows=json.loads((ROOT/'waveforms.json').read_text())
modes=load_beam_modes(ROOT,rows)
index=json.loads((ROOT/'plot_index.json').read_text())
grouped=collections.defaultdict(lambda:collections.defaultdict(list))
for cycle,records in rows.items():
    for row_index,record in enumerate(records):grouped[record['pv']][cycle].append(row_index)
cache={cycle:np.memmap(ROOT/'cache'/f'{cycle}.f32',dtype='<f4',mode='r',shape=(len(records),BINS*2)) for cycle,records in rows.items()}
small=font(10);label=font(12);heading=font(15);titlefont=font(23)
panelw,panelh,columns,titleh=360,220,6,102
left,top,right,bottom=60,43,345,183
manifest=[];filenames=set();started=time.time()
for number,pv in enumerate(sorted(grouped),1):
    cycles=sorted(grouped[pv],key=int)
    selected=[rows[c][i] for c in cycles for i in grouped[pv][c]]
    finite=[r for r in selected if r['min'] is not None]
    low=min(r['min'] for r in finite) if finite else -1
    high=max(r['max'] for r in finite) if finite else 1
    if high==low:high=low+1
    pad=(high-low)*0.04;low-=pad;high+=pad
    maxsamples=max(1,max(r['samples'] for r in selected)-1)
    template=Image.new('RGB',(panelw,panelh),'white');d=ImageDraw.Draw(template)
    for k in range(4):
        y=top+(bottom-top)*k/3;x=left+(right-left)*k/3
        d.line((left,y,right,y),fill='#e3e9ef');d.line((x,top,x,bottom),fill='#e3e9ef')
        d.text((4,y-6),f'{high-(high-low)*k/3:.3g}',font=small,fill='#435365')
        d.text((x-12,bottom+5),f'{maxsamples*k/3:.0f}',font=small,fill='#435365')
    d.line((left,top,left,bottom,right,bottom),fill='#435365')
    d.text((150,205),'Sample index',font=small,fill='#435365')
    ncols=min(columns,len(cycles));nrows=math.ceil(len(cycles)/ncols)
    canvas=Image.new('RGB',(ncols*panelw,nrows*panelh+titleh),'#e8edf3')
    draw=ImageDraw.Draw(canvas);draw.text((15,12),pv,font=titlefont,fill='#152536')
    draw.text((15,45),f'{len(cycles)} cycle_id panels | {len(selected):,} rows | {len(finite):,} nonempty waveforms',font=heading,fill='#435365')
    draw.text((15,70),'Shared axes within this PV; raw amplitude; 256-bin min/max envelope; duplicates included.',font=label,fill='#435365')
    trace=ImageDraw.Draw(canvas,'RGBA')
    color=tuple(bytes.fromhex(index['pvs'][pv][1:]))
    for j,cycle in enumerate(cycles):
        ox=(j%ncols)*panelw;oy=(j//ncols)*panelh+titleh
        canvas.paste(template,(ox,oy))
        selected_indices=grouped[pv][cycle]
        draw.text((ox+10,oy+6),f'cycle_id: {cycle}',font=heading,fill='#152536')
        nonempty=sum(rows[cycle][i]['min'] is not None for i in selected_indices)
        mode=panel_mode(modes,cycle,selected_indices)
        draw.text((ox+10,oy+24),mode_caption(mode,nonempty,len(selected_indices)),font=label,fill='#175c47')
        for i in selected_indices:
            record=rows[cycle][i]
            if record['min'] is None:continue
            values=cache[cycle][i]
            x=ox+left+np.repeat((np.arange(BINS)+0.5)/BINS,2)*(record['samples']-1)/maxsamples*(right-left)
            y=oy+bottom-(values-low)/(high-low)*(bottom-top)
            points=[(float(a),float(b)) for a,b in zip(x,y) if np.isfinite(b)]
            if len(points)>1:trace.line(points,fill=(*color,200),width=1)
    filename=re.sub(r'[^A-Za-z0-9_.-]+','_',pv)+'.png'
    assert filename not in filenames;filenames.add(filename)
    canvas.save(OUT/filename,compress_level=1)
    manifest.append({'pv':pv,'file':filename,'cycles':len(cycles),'rows':len(selected),
                     'nonempty_waveforms':len(finite),'width':canvas.width,'height':canvas.height})
    if number%20==0 or number==len(grouped):print(f'[{number}/{len(grouped)}] PNGs saved; {time.time()-started:.0f}s',flush=True)
    del trace,draw,canvas

assert sum(r['rows'] for r in manifest)==sum(map(len,rows.values()))
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
with (OUT/'pv_files.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(manifest[0]));writer.writeheader();writer.writerows(manifest)
items='\n'.join(f'<tr><td><a href="{html.escape(r["file"],quote=True)}">{html.escape(r["pv"])}</a></td><td>{r["cycles"]}</td><td>{r["rows"]}</td><td>{r["nonempty_waveforms"]}</td></tr>' for r in manifest)
page='''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>PV別の波形</title><style>body{font:15px system-ui;margin:30px;color:#193044;background:#f5f8fc}table{border-collapse:collapse;background:white;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #d5dfe9}a{color:#126092}input{padding:10px;width:400px;max-width:90%}p{line-height:1.7}</style><h1>PV別の波形 — 521ファイル</h1><p>各PVにつき1枚のPNGです。cycle_idごとにパネルを分け、運転モード（beam_mode）を表示しています。同じPV内では縦軸・横軸の範囲を共通にしています。波形は256区間の最小値・最大値で表示しています。全218,533行が対象で、数値のある波形は218,527本です。</p><p><a href="pv_files.csv" download>ファイル一覧CSV</a> · <a href="../index.html">全PVを重ねたサイクル別表示</a></p><input id="filter" placeholder="PV名で検索"><table><thead><tr><th>PV名（画像へのリンク）</th><th>サイクル数</th><th>行数</th><th>数値のある波形</th></tr></thead><tbody>ITEMS</tbody></table><script>document.getElementById('filter').oninput=function(){const q=this.value.toLowerCase();for(const r of document.querySelectorAll('tbody tr'))r.hidden=!r.firstElementChild.textContent.toLowerCase().includes(q)}</script></html>'''
(OUT/'index.html').write_text(page.replace('ITEMS',items),encoding='utf-8')
print('Complete:',OUT,flush=True)
