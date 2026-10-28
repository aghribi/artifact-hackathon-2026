"""Expose existing PV plots by their source Parquet group."""
from pathlib import Path
import collections, csv, html, json
from PIL import Image

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'by_pv'
rows=json.loads((ROOT/'waveforms.json').read_text())
manifest=json.loads((OUT/'manifest.json').read_text())
files={r['pv']:r for r in manifest}
groups=collections.defaultdict(lambda:collections.defaultdict(lambda:{'rows':0,'nonempty':0,'cycles':set()}))
for cycle,records in rows.items():
    for r in records:
        item=groups[Path(r['member']).stem][r['pv']]
        item['rows']+=1;item['nonempty']+=r['min'] is not None;item['cycles'].add(cycle)

style='<style>body{font:15px system-ui;margin:30px;color:#193044;background:#f5f8fc}table{border-collapse:collapse;background:white;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #d5dfe9}a{color:#126092}input{padding:10px;width:400px;max-width:90%}p{line-height:1.7}img{width:100%;max-width:1600px}</style>'
search="<script>document.getElementById('filter').oninput=function(){const q=this.value.toLowerCase();for(const r of document.querySelectorAll('tbody tr'))r.hidden=!r.firstElementChild.textContent.toLowerCase().includes(q)}</script>"
summary=[]
for group in ['MBL','SPOKES','BCM']:
    items=groups[group];ordered=sorted(items)
    assert all((OUT/files[pv]['file']).is_file() for pv in ordered)
    count=sum(v['rows'] for v in items.values());nonempty=sum(v['nonempty'] for v in items.values())
    representative=ordered[0]
    with Image.open(OUT/files[representative]['file']) as image:
        image.crop((0,0,image.width,min(image.height,322))).save(OUT/f'{group}_example.png')
    table='\n'.join(f'<tr><td><a href="{html.escape(files[pv]["file"],quote=True)}">{html.escape(pv)}</a></td><td>{len(items[pv]["cycles"])}</td><td>{items[pv]["rows"]}</td><td>{items[pv]["nonempty"]}</td></tr>' for pv in ordered)
    page=f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{group} のPV別波形</title>{style}<h1>{group} — PV別の波形</h1><p>{len(items)}種類のPV・{count:,}行・数値のある波形 {nonempty:,}本。各PVに1枚のPNGを用意し、cycle_idごとにパネルを分け、運転モード（beam_mode）を表示しています。同じPV内で軸の範囲は共通です。描画は256区間の最小値・最大値を使用し、dd.listの重複も含みます。</p><p><a href="non_hbl.html">HBL以外の一覧</a> · <a href="{group}_files.csv">このグループのCSV</a> · <a href="index.html">全PVの一覧</a></p><p>代表例（最初のパネル行のみ）。全サイクルはPV名のリンクから開けます。</p><a href="{html.escape(files[representative]['file'],quote=True)}"><img src="{group}_example.png" alt="{html.escape(representative,quote=True)}"></a><p><input id="filter" placeholder="PV名で検索"></p><table><thead><tr><th>PV名（全サイクルの画像）</th><th>サイクル数</th><th>行数</th><th>数値のある波形</th></tr></thead><tbody>{table}</tbody></table>{search}</html>'''
    (OUT/f'{group}.html').write_text(page,encoding='utf-8')
    with (OUT/f'{group}_files.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['pv','file','cycles','rows','nonempty_waveforms'])
        for pv in ordered:w.writerow([pv,files[pv]['file'],len(items[pv]['cycles']),items[pv]['rows'],items[pv]['nonempty']])
    summary.append({'group':group,'pvs':len(items),'rows':count,'nonempty':nonempty})
links=''.join(f'<tr><td><a href="{r["group"]}.html">{r["group"]}</a></td><td>{r["pvs"]}</td><td>{r["rows"]:,}</td><td>{r["nonempty"]:,}</td></tr>' for r in summary)
(OUT/'non_hbl.html').write_text(f'<!doctype html><html lang="ja"><meta charset="utf-8"><title>HBL以外の波形</title>{style}<h1>HBL以外のPV別波形</h1><p>グループを選ぶとPV別のPNGを開けます。各PNGではcycle_idごとに別パネルを表示します。</p><table><tr><th>グループ</th><th>PV数</th><th>行数</th><th>数値のある波形</th></tr>{links}</table><p><a href="index.html">全PVの一覧</a></p></html>',encoding='utf-8')
index=OUT/'index.html';page=index.read_text()
if 'non_hbl.html' not in page:
    page=page.replace('<input id="filter"','<p><a href="non_hbl.html">HBL以外のグループ別表示（MBL・SPOKES・BCM）</a></p><input id="filter"',1)
    index.write_text(page,encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
