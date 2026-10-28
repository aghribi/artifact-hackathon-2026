import json, csv
from pathlib import Path

OUT=Path(__file__).resolve().parent
index=json.loads((OUT/'plot_index.json').read_text())
rows=json.loads((OUT/'waveforms.json').read_text())
index['total_rows']=sum(len(v) for v in rows.values())
index['empty_rows']=sum(r['min'] is None for v in rows.values() for r in v)
for cycle in index['cycles']:
    cycle['nonempty']=sum(r['min'] is not None for r in rows[str(cycle['id'])])
with (OUT/'cycle_summary.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['cycle_id','rows','nonempty_waveforms','unique_pvs'])
    for c in index['cycles']:w.writerow([c['id'],c['waveforms'],c['nonempty'],len(c['pvs'])])
html='''<!doctype html>
<html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>全PVの波形 — cycle_id別</title>
<style>
body{font:15px system-ui,sans-serif;margin:0;background:#edf2f7;color:#193044}header{padding:22px 30px;background:#fff;border-bottom:1px solid #ccd6e0}h1{font-size:25px;margin:0 0 12px}p{line-height:1.65;margin:8px 0}main{padding:22px 30px}button,select,input{padding:8px;border:1px solid #b5c5d3;border-radius:5px;background:white;color:#193044}button{cursor:pointer}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:14px}.detail{display:grid;grid-template-columns:minmax(0,1fr) 420px;gap:16px}.detail img{width:100%;background:white}.legend{height:500px;overflow:auto;background:white;padding:12px}.legend div{font-size:12px;white-space:nowrap;padding:5px}.swatch{display:inline-block;width:20px;height:4px;margin-right:7px;vertical-align:middle}#overview{overflow:auto;max-height:80vh;background:#d5dfe9;border:1px solid #b5c5d3}#overview img{display:block;max-width:none;cursor:crosshair}.hint{font-size:13px;color:#526879}a{color:#126092}@media(max-width:900px){.detail{grid-template-columns:1fr}.legend{height:220px}}
</style>
<header><h1>全PVの波形 — cycle_id ごとのプロット</h1><p id="summary"></p><p>各パネルに同じ cycle_id の全PVを重ねています。PVの色は全パネルで共通です。横軸はサンプル番号、縦軸は元の信号値で、縦軸の範囲はサイクルごとに調整しています。</p><p class="hint">全44アーカイブが対象（重複を含む）。表示には256区間の最小値・最大値を使用。数値のない行は曲線を描かず、件数に残しています。</p></header>
<main><p><a href="by_pv/index.html">PV別の画像ファイル一覧</a></p><div class="controls"><label>cycle_id <select id="cycle"></select></label><button id="previous">前へ</button><button id="next">次へ</button><a id="panelDownload" download>このサイクルの画像</a><a href="all_cycles.png" download>全サイクルを1枚のPNGで保存</a><a href="cycle_summary.csv" download>サイクル別件数CSV</a></div>
<div class="detail"><img id="detailImage" alt="選択サイクルの全PV波形"><div class="legend"><b>PVと色</b><div id="legend"></div></div></div>
<h2>全737サイクルを1枚で表示</h2><p class="hint">画像内のパネルをクリックすると、上の詳細画像とPV一覧が切り替わります。最初は閲覧用の縮小画像を表示します。</p><div class="controls"><label>表示幅 <input id="zoom" type="range" min="600" max="9600" step="100" value="1600"></label><button id="fit">画面幅に合わせる</button><button id="fullResolution">高解像度に切替</button><span id="zoomLabel"></span></div>
<div id="overview"><img id="allImage" src="all_cycles_preview.jpg" alt="全サイクルの波形一覧"></div></main>
<script>
const index=INDEX_DATA;
const select=document.getElementById('cycle');
for(const c of index.cycles){const o=document.createElement('option');o.value=c.id;o.textContent=c.id+' ('+c.nonempty+'波形 / '+c.pvs.length+'PV)';select.append(o)}
document.getElementById('summary').textContent=index.cycles.length+'サイクル・'+Object.keys(index.pvs).length+'種類のPV・'+index.total_rows.toLocaleString()+'行（うち数値のない行 '+index.empty_rows.toLocaleString()+'）';
function show(){const c=index.cycles[select.selectedIndex];const path='cycles/'+c.id+'.png';document.getElementById('detailImage').src=path;document.getElementById('panelDownload').href=path;const box=document.getElementById('legend');box.replaceChildren();for(const pv of c.pvs){const row=document.createElement('div');const swatch=document.createElement('span');swatch.className='swatch';swatch.style.background=index.pvs[pv];row.append(swatch,document.createTextNode(pv));box.append(row)}}
select.onchange=show;document.getElementById('previous').onclick=()=>{select.selectedIndex=Math.max(0,select.selectedIndex-1);show()};document.getElementById('next').onclick=()=>{select.selectedIndex=Math.min(index.cycles.length-1,select.selectedIndex+1);show()};show();
const all=document.getElementById('allImage'),zoom=document.getElementById('zoom');function scale(width){all.style.width=width+'px';document.getElementById('zoomLabel').textContent=width+' px'}zoom.oninput=()=>scale(zoom.value);document.getElementById('fit').onclick=()=>scale(document.getElementById('overview').clientWidth);scale(1600);
document.getElementById('fullResolution').onclick=()=>{all.src='all_cycles.png';scale(4800);zoom.value=4800};
all.onclick=e=>{const rect=all.getBoundingClientRect(),x=(e.clientX-rect.left)/rect.width*4800,y=(e.clientY-rect.top)/rect.height*(Math.ceil(index.cycles.length/8)*340+110);if(y<110)return;const i=Math.floor((y-110)/340)*8+Math.floor(x/600);if(i<index.cycles.length){select.selectedIndex=i;show();document.getElementById('detailImage').scrollIntoView({behavior:'smooth',block:'center'})}};
</script></html>'''
(OUT/'index.html').write_text(html.replace('INDEX_DATA',json.dumps(index,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
print('Viewer created:',OUT/'index.html')
print('Nonempty waveforms:',index['total_rows']-index['empty_rows'],'Empty/nonfinite rows:',index['empty_rows'])
