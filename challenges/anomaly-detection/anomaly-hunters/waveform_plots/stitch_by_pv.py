"""Concatenate the existing by_pv PNGs into one readable-size mosaic per group."""
import csv, functools, html, json, math, textwrap, time
from pathlib import Path
from PIL import Image, ImageDraw
from plot_all_cycles import font

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'by_pv'
OUT=SOURCE/'combined';OUT.mkdir(exist_ok=True)
font=functools.lru_cache(maxsize=16)(font)
manifest=json.loads((SOURCE/'manifest.json').read_text())
by_name={r['pv']:r for r in manifest}
groups={}
for group in ['MBL','SPOKES','BCM']:
    with (SOURCE/f'{group}_files.csv').open() as f:
        groups[group]={r['pv'] for r in csv.DictReader(f)}
others=set().union(*groups.values());groups['HBL']=set(by_name)-others
assert sum(map(len,groups.values()))==len(manifest)==521

inventory=json.loads((ROOT/'inventory.json').read_text())
archive_paths=list(dict.fromkeys(r['archive'] for r in inventory['files']))
archive_ids={path:f'A{i:03d}' for i,path in enumerate(archive_paths,1)}
pv_archives={pv:set() for pv in by_name}
with (OUT/'sources.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['archive_id','tar_file','tar_path'])
    for path in archive_paths:w.writerow([archive_ids[path],Path(path).name,path])
with (OUT/'pv_cycle_sources.csv').open('w',newline='') as f, (OUT/'parquet_files.csv').open('w',newline='') as q:
    w=csv.writer(f);w.writerow(['group','pv','cycle_id','archive_id','tar_file','tar_path','parquet_path','parquet_row_index_zero_based','plot_png'])
    qw=csv.writer(q);qw.writerow(['archive_id','tar_file','parquet_path','rows'])
    for r in inventory['files']:
        path=r['archive'];aid=archive_ids[path];group=Path(r['member']).stem
        qw.writerow([aid,Path(path).name,r['member'],r['rows']])
        for rowindex,(pv,cycle) in enumerate(zip(r['pvs'],r['cycle_ids'])):
            pv_archives[pv].add(aid)
            w.writerow([group,pv,cycle,aid,Path(path).name,path,r['member'],rowindex,'../'+by_name[pv]['file']])

def ranges(ids):
    numbers=sorted(int(a[1:]) for a in ids);blocks=[]
    first=last=numbers[0]
    for n in numbers[1:]:
        if n==last+1:last=n;continue
        blocks.append(f'A{first:03d}'+(f'-A{last:03d}' if first!=last else ''));first=last=n
    blocks.append(f'A{first:03d}'+(f'-A{last:03d}' if first!=last else ''))
    return ', '.join(blocks)

TITLE=160;GAP=8;LABEL=120;PIXELS=120_000_000;MAX_W=8500;MAX_H=28000
def layout(records,columns,tilewidth):
    heights=[TITLE]*columns;positions=[]
    for r in records:
        col=min(range(columns),key=lambda j:heights[j])
        imageheight=round(r['height']*tilewidth/r['width'])
        positions.append((col*(tilewidth+GAP)+GAP,heights[col],imageheight))
        heights[col]+=LABEL+imageheight+GAP
    return columns*(tilewidth+GAP)+GAP,max(heights)+GAP,positions

def choose_layout(records):
    candidates=[]
    for columns in range(1,min(32,len(records))+1):
        lo,hi=60,min(1500,(MAX_W-GAP)//columns-GAP)
        best=None
        while lo<=hi:
            mid=(lo+hi)//2;w,h,pos=layout(records,columns,mid)
            if w*h<=PIXELS and w<=MAX_W and h<=MAX_H:
                best=(mid,w,h,pos);lo=mid+1
            else:hi=mid-1
        if best:
            mid,w,h,pos=best
            candidates.append((mid,-abs(w/h-0.3),columns,w,h,pos))
    assert candidates
    return max(candidates,key=lambda c:(c[0],c[1]))

summary=[];started=time.time()
for group in ['HBL','MBL','SPOKES','BCM']:
    records=[by_name[pv] for pv in sorted(groups[group])]
    tilewidth,_,columns,width,height,positions=choose_layout(records)
    print(f'{group}: {len(records)} source PNGs -> {width}x{height}, {columns} columns, tile width {tilewidth}',flush=True)
    canvas=Image.new('RGB',(width,height),'#e8edf3');draw=ImageDraw.Draw(canvas)
    draw.text((20,15),f'{group}: existing by_pv plots stitched into one image',font=font(32),fill='#152536')
    total=sum(r['rows'] for r in records)
    draw.text((20,58),f'{len(records)} PV image files | {total:,} rows | cycle_id and beam_mode labels preserved',font=font(19),fill='#435365')
    draw.text((20,87),'Overview uses resized copies. Open the original PV PNG for readable cycle labels and full plot resolution.',font=font(15),fill='#435365')
    draw.text((20,114),'Archive IDs A###: sources.csv. Exact PV / cycle_id / Parquet row: pv_cycle_sources.csv.',font=font(15),fill='#435365')
    for number,(r,(x,y,imageheight)) in enumerate(zip(records,positions),1):
        labelsize=13 if tilewidth>=300 else 11
        chars=max(20,int(tilewidth/(labelsize*0.6)))
        lines=textwrap.wrap(r['pv'],width=chars,break_long_words=True,break_on_hyphens=False)
        for k,line in enumerate(lines[:4]):draw.text((x+4,y+3+k*(labelsize+2)),line,font=font(labelsize),fill='#152536')
        source_lines=textwrap.wrap('Tar: '+ranges(pv_archives[r['pv']]),width=max(20,int(tilewidth/6.0)))
        for k,line in enumerate(source_lines):draw.text((x+4,y+65+k*13),line,font=font(11),fill='#435365')
        draw.text((x+4,y+LABEL-20),f'{r["cycles"]} cycles / {r["rows"]:,} rows',font=font(11),fill='#435365')
        with Image.open(SOURCE/r['file']) as image:
            assert image.size==(r['width'],r['height'])
            thumbnail=image.resize((tilewidth,imageheight),Image.Resampling.LANCZOS,reducing_gap=3.0)
        canvas.paste(thumbnail,(x,y+LABEL))
        del thumbnail
        if number%40==0:print(f'{group}: {number}/{len(records)} stitched ({time.time()-started:.0f}s)',flush=True)
    canvas.save(OUT/f'{group}.png',compress_level=3)
    canvas.thumbnail((2400,16000));canvas.save(OUT/f'{group}_preview.jpg',quality=85)
    summary.append({'group':group,'file':f'{group}.png','pv_files':len(records),'rows':total,
                    'width':width,'height':height,'columns':columns,'tile_width':tilewidth,
                    'sources':[r['file'] for r in records],
                    'pv_archive_ids':{r['pv']:sorted(pv_archives[r['pv']]) for r in records}})
    del draw,canvas
    print(f'Saved {group}.png ({time.time()-started:.0f}s)',flush=True)

assert sum(r['rows'] for r in summary)==218533
(OUT/'manifest.json').write_text(json.dumps(summary,indent=2))
links=''.join(f'<tr><td>{r["group"]}</td><td>{r["pv_files"]}</td><td><a href="{r["group"]}.png">連結PNG</a></td><td><a href="{r["group"]}_preview.jpg">軽量JPEG</a></td><td>{r["width"]} × {r["height"]}</td></tr>' for r in summary)
(OUT/'index.html').write_text(f'<!doctype html><html lang="ja"><meta charset="utf-8"><title>PV別画像のグループ別連結</title><style>body{{font:16px system-ui;margin:30px;color:#193044}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #ddd}}a{{color:#126092}}</style><h1>PV別PNGをグループごとに1枚へ連結</h1><p>by_pv の既存PNGをそのまま使用し、各PV内のcycle_id別パネルと運転モード（beam_mode）を保持しています。連結画像は縮小して配置しています。細部は<a href="../index.html">元のPV別PNG</a>で確認できます。</p><p>画像内のA番号は元tarの参照番号です。<a href="sources.csv">A番号と元tarファイル名</a> · <a href="pv_cycle_sources.csv">PV・cycle_id・元tar・Parquetパス・行番号の対応表</a> · <a href="parquet_files.csv">Parquetファイル一覧</a></p><table><tr><th>グループ</th><th>PV画像数</th><th>PNG</th><th>JPEG</th><th>画像サイズ</th></tr>{links}</table></html>',encoding='utf-8')
print('All mosaics complete.',flush=True)
