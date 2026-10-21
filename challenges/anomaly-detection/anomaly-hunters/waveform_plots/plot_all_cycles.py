import collections, json, tarfile, struct
import math, colorsys, argparse, io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from parquet_metadata import Compact
from parquet_reader import column, decompress

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent

def read_column(stream, metadata):
    start=metadata.get(11,metadata[9])
    stream.seek(start)
    raw=stream.read(metadata[7])
    assert len(raw)==metadata[7]
    local=dict(metadata)
    for key in (9,11):
        if key in local:local[key]-=start
    return column(raw,{str(k):v for k,v in local.items()})

def archives():
    for line in (ROOT/'dd.list').read_text().splitlines():
        if line.strip():yield Path(line.strip().replace('/workspace/lustre/','/lustre/',1))

def inventory():
    records=[];cycles=collections.Counter();pvs=collections.Counter()
    for archive_path in archives():
        with tarfile.open(archive_path,'r:*') as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith('.parquet'):continue
                stream=archive.extractfile(member)
                stream.seek(member.size-8);tail=stream.read(8)
                assert tail[-4:]==b'PAR1'
                size=struct.unpack('<I',tail[:4])[0]
                stream.seek(member.size-8-size);meta=Compact(stream.read(size)).obj()
                row_cycles=[];row_pvs=[]
                for group in meta[4]:
                    columns={c[3][3][0]:c[3] for c in group[1]}
                    ids=read_column(stream,columns['cycle_id']);names=read_column(stream,columns['pv'])
                    assert len(ids)==len(names)==group[3]
                    row_cycles.extend(ids);row_pvs.extend(names)
                cycles.update(row_cycles);pvs.update(row_pvs)
                records.append({'archive':str(archive_path),'member':member.name,'rows':meta[3],
                                'metadata':meta,'cycle_ids':row_cycles,'pvs':row_pvs})
    report={'cycles':dict(cycles),'pvs':dict(pvs),'files':records}
    (OUT/'inventory.json').write_text(json.dumps(report))
    print('FILES',len(records),'ROWS',sum(cycles.values()),'CYCLES',len(cycles),'PVS',len(pvs),flush=True)
    print('WAVEFORMS PER CYCLE',min(cycles.values()),max(cycles.values()),flush=True)
    return report

def levels(raw,width,count):
    reader=Compact(raw); result=np.empty(count,dtype=np.uint8); pos=0
    while pos<count:
        header=reader.varint()
        if header&1:
            n=(header>>1)*8
            bits=np.unpackbits(np.frombuffer(reader.f.read(n*width//8),dtype=np.uint8),bitorder='little')
            vals=(bits.reshape(n,width)*(1<<np.arange(width))).sum(axis=1)
            take=min(n,count-pos);result[pos:pos+take]=vals[:take];pos+=take
        else:
            n=header>>1;v=int.from_bytes(reader.f.read((width+7)//8),'little')
            take=min(n,count-pos);result[pos:pos+take]=v;pos+=take
    return result

def waveforms(stream,m):
    offset=m.get('11',m['9']); end=offset+m['7'];pending=[]
    assert m['1']==4 and m['4']==6
    while offset<end:
        stream.seek(offset);reader=Compact(stream.read(min(16384,end-offset)))
        header=reader.obj();offset+=reader.f.tell()
        stream.seek(offset);raw=stream.read(header[3]);offset+=header[3]
        assert header[1]==0 and header[5][2]==0,header
        payload=decompress(raw,header[2]);f=io.BytesIO(payload);count=header[5][1]
        n=struct.unpack('<I',f.read(4))[0];reps=levels(f.read(n),1,count)
        n=struct.unpack('<I',f.read(4))[0];defs=levels(f.read(n),2,count)
        stored=np.frombuffer(f.read(),dtype='<f4');assert len(stored)==np.count_nonzero(defs==3)
        if np.all(defs==3):values=stored
        else:
            values=np.full(count,np.nan,dtype=np.float32);values[defs==3]=stored
        def segment(a,b):return values[a:b][defs[a:b]>=2] if np.any(defs[a:b]<2) else values[a:b]
        starts=np.flatnonzero(reps==0)
        if len(starts)==0:pending.append(segment(0,count));continue
        if starts[0]>0:pending.append(segment(0,starts[0]))
        if pending:
            yield np.concatenate(pending);pending=[]
        for a,b in zip(starts[:-1],starts[1:]):yield segment(a,b)
        pending=[segment(starts[-1],count)]
    if pending:yield np.concatenate(pending)

BINS=256
def envelope(values):
    n=len(values)
    if n==0:return np.full(BINS*2,np.nan,dtype='<f4')
    if n<BINS:
        return np.interp(np.linspace(0,n-1,BINS*2),np.arange(n),values).astype('<f4')
    edges=np.linspace(0,n,BINS+1,dtype=np.int64)
    lo=np.minimum.reduceat(values,edges[:-1]);hi=np.maximum.reduceat(values,edges[:-1])
    return np.column_stack((lo,hi)).astype('<f4').ravel()

def cache_waveforms(report):
    cache=OUT/'cache';cache.mkdir(exist_ok=True)
    rows=collections.defaultdict(list)
    handles={}
    files=report['files']; grouped=collections.defaultdict(list)
    for record in files:grouped[record['archive']].append(record)
    done=0
    try:
        for archive_index,(path,records) in enumerate(grouped.items(),1):
            with tarfile.open(path,'r:*') as archive:
                for record in records:
                    stream=archive.extractfile(record['member']);position=0
                    for group in record['metadata']['4']:
                        m=next(c['3'] for c in group['1'] if c['3']['3'][0]=='data_array')
                        count=0
                        for values in waveforms(stream,m):
                            cycle=record['cycle_ids'][position];pv=record['pvs'][position]
                            if cycle not in handles:handles[cycle]=(cache/f'{cycle}.f32').open('wb')
                            sampled=envelope(values);handles[cycle].write(sampled.tobytes())
                            finite=values[np.isfinite(values)]
                            rows[cycle].append({'pv':pv,'samples':len(values),'min':float(finite.min()) if len(finite) else None,
                                                'max':float(finite.max()) if len(finite) else None,'archive':archive_index,
                                                'member':record['member']})
                            position+=1;count+=1
                        assert count==group['3'],(record['member'],count,group['3'])
                    assert position==record['rows']
                    done+=position
            print(f'[{archive_index}/{len(grouped)}] decoded {done:,} waveforms',flush=True)
    finally:
        for h in handles.values():h.close()
    (OUT/'waveforms.json').write_text(json.dumps(rows,ensure_ascii=False))
    assert done==sum(report['cycles'].values())
    return rows

def font(size):
    return ImageFont.truetype('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf',size)

def render_panel(cycle,records,data,colors,size):
    w,h=size;im=Image.new('RGB',size,'white');d=ImageDraw.Draw(im)
    left,top,right,bottom=80,62,w-25,h-55
    valid=[r for r in records if r['min'] is not None]
    low=min(r['min'] for r in valid) if valid else -1;high=max(r['max'] for r in valid) if valid else 1
    if high==low:high=low+1
    padding=(high-low)*0.04;low-=padding;high+=padding
    maxsamples=max(1,max(r['samples'] for r in records))
    d.text((15,10),f'cycle_id: {cycle}',font=font(21),fill='#152536')
    d.text((15,36),f'{len(records)} rows / {len(valid)} nonempty waveforms / {len(set(r["pv"] for r in records))} PVs',font=font(14),fill='#566675')
    for i in range(5):
        y=top+(bottom-top)*i/4;v=high-(high-low)*i/4
        d.line((left,y,right,y),fill='#e5e9ef')
        d.text((4,y-8),f'{v:.3g}',font=font(13),fill='#435365')
        x=left+(right-left)*i/4
        d.line((x,top,x,bottom),fill='#e5e9ef')
        d.text((x-12,bottom+6),f'{maxsamples*i/4:.0f}',font=font(12),fill='#435365')
    trace=ImageDraw.Draw(im,'RGBA')
    for record,values in zip(records,data):
        x=left+np.repeat((np.arange(BINS)+0.5)/BINS,2)*(record['samples']-1)/maxsamples*(right-left)
        y=bottom-(values-low)/(high-low)*(bottom-top)
        points=[(float(a),float(b)) for a,b in zip(x,y) if np.isfinite(b)]
        if len(points)>1:trace.line(points,fill=(*colors[record['pv']],100),width=1)
    d.line((left,top,left,bottom,right,bottom),fill='#435365',width=1)
    d.text(((left+right)//2-70,h-24),'Sample index',font=font(14),fill='#435365')
    return im

def render(report,rows):
    panels=OUT/'cycles';panels.mkdir(exist_ok=True)
    pvnames=sorted(report['pvs'])
    colors={pv:tuple(round(v*255) for v in colorsys.hsv_to_rgb((i*0.61803398875)%1,0.75,0.70)) for i,pv in enumerate(pvnames)}
    cycles=sorted(rows,key=int);columns=8;size=(600,340);nrows=math.ceil(len(cycles)/columns)
    titleheight=110
    overview=Image.new('RGB',(columns*size[0],nrows*size[1]+titleheight),'#e8edf3')
    d=ImageDraw.Draw(overview)
    d.text((25,18),'All PV waveforms, separated by cycle_id',font=font(32),fill='#152536')
    d.text((25,61),f'{len(cycles)} cycles | {len(pvnames)} PVs | {sum(len(v) for v in rows.values()):,} waveforms | dd.list (duplicates included)',font=font(18),fill='#435365')
    d.text((25,86),'Raw amplitude; sample-index x axis; independent y scale per cycle; 256-bin min/max envelope.',font=font(15),fill='#435365')
    for i,cycle in enumerate(cycles):
        records=rows[cycle];data=np.memmap(OUT/'cache'/f'{cycle}.f32',dtype='<f4',mode='r',shape=(len(records),BINS*2))
        panel=render_panel(cycle,records,data,colors,size)
        overview.paste(panel,((i%columns)*size[0],(i//columns)*size[1]+titleheight))
        large=render_panel(cycle,records,data,colors,(1100,560));large.save(panels/f'{cycle}.png')
        if (i+1)%100==0:print(f'Rendered {i+1}/{len(cycles)} cycles',flush=True)
    overview.save(OUT/'all_cycles.png',compress_level=3)
    overview.thumbnail((2400,16000));overview.save(OUT/'all_cycles_preview.jpg',quality=85)
    index={'cycles':[{'id':cycle,'waveforms':len(rows[cycle]),'pvs':sorted(set(r['pv'] for r in rows[cycle]))} for cycle in cycles],
           'pvs':{pv:'#%02x%02x%02x'%colors[pv] for pv in pvnames}}
    (OUT/'plot_index.json').write_text(json.dumps(index))
    print('Saved overview',OUT/'all_cycles.png',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--inventory-only',action='store_true');args=parser.parse_args()
    report=json.loads((OUT/'inventory.json').read_text()) if (OUT/'inventory.json').exists() else inventory()
    if not args.inventory_only:
        rows=json.loads((OUT/'waveforms.json').read_text()) if (OUT/'waveforms.json').exists() else cache_waveforms(report)
        render(report,rows)
