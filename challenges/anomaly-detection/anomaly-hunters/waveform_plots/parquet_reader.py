import sys, json, struct, ctypes, io, collections
from pathlib import Path
import numpy as np
from parquet_metadata import Compact

zstd=ctypes.CDLL('libzstd.so.1')
zstd.ZSTD_decompress.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t]
zstd.ZSTD_decompress.restype=ctypes.c_size_t
def decompress(raw,n):
    out=ctypes.create_string_buffer(n)
    size=zstd.ZSTD_decompress(out,n,raw,len(raw))
    assert size==n,(size,n)
    return out.raw
def rle(raw,width,count):
    f=Compact(raw); out=[]
    while len(out)<count:
        h=f.varint()
        if h&1:
            n=(h>>1)*8; b=f.f.read(n*width//8)
            bits=int.from_bytes(b,'little'); mask=(1<<width)-1
            out.extend((bits>>(i*width))&mask for i in range(n))
        else:
            v=int.from_bytes(f.f.read((width+7)//8),'little');out.extend([v]*(h>>1))
    return out[:count]
def plain(raw,t,n):
    if t==6:
        out=[];f=io.BytesIO(raw)
        for _ in range(n):
            length=struct.unpack('<I',f.read(4))[0];out.append(f.read(length).decode())
        return out
    if t==0:return [(raw[i//8]>>(i%8))&1==1 for i in range(n)]
    return np.frombuffer(raw,dtype={1:'<u4',2:'<u8',4:'<f4',5:'<f8'}[t],count=n).tolist()
def column(data,m):
    offset=m.get('11',m['9']); end=offset+m['7']; dictionary=None;out=[];current=None
    islist=len(m['3'])>1; maxdef=3 if islist else 1
    while offset<end:
        p=Compact(data[offset:]);h=p.obj();offset+=p.f.tell()
        raw=data[offset:offset+h[3]];offset+=h[3]
        assert m['4']==6
        payload=decompress(raw,h[2])
        if h[1]==2:
            dictionary=plain(payload,m['1'],h[7][1]);continue
        assert h[1]==0,h
        header=h[5]; count=header[1];enc=header[2];f=io.BytesIO(payload)
        levels=[]
        if islist:
            n=struct.unpack('<I',f.read(4))[0];reps=rle(f.read(n),1,count)
        else:reps=[0]*count
        n=struct.unpack('<I',f.read(4))[0];defs=rle(f.read(n),maxdef.bit_length(),count)
        valid=sum(d==maxdef for d in defs);values=f.read()
        if enc==0:vals=plain(values,m['1'],valid)
        elif enc in (2,8):vals=[dictionary[i] for i in rle(values[1:],values[0],valid)]
        else:raise ValueError(enc)
        it=iter(vals)
        for rep,d in zip(reps,defs):
            v=next(it) if d==maxdef else None
            if not islist:out.append(v)
            else:
                if rep==0:
                    current=[] if d>=1 else None;out.append(current)
                if d>=2:current.append(v)
    return out

if __name__=='__main__':
    summaries=json.loads(Path('data/dd_inspected/metadata.json').read_text())
    results=[]
    for s in summaries:
        if not s['path']:continue
        data=Path(s['path']).read_bytes();cols={}
        for rg in s['metadata']['4']:
            for c in rg['1']:
                m=c['3'];name=m['3'][0];cols.setdefault(name,[]).extend(column(data,m))
        assert all(len(v)==s['metadata']['3'] for v in cols.values())
        print(s['member'],'rows',s['metadata']['3'])
        row={k:(v[0][:10] if k=='data_array' else v[0]) for k,v in cols.items()}
        print('FIRST ROW',json.dumps(row,ensure_ascii=False))
        counts=dict(collections.Counter(cols['pv']));lengths=dict(collections.Counter(map(len,cols['data_array'])))
        print('PV counts',json.dumps(counts));print('waveform lengths',lengths)
        print('event_id unique',len(set(cols['event_id'])),'cycle_id unique',len(set(cols['cycle_id'])))
        print('nulls',{k:sum(x is None for x in v) for k,v in cols.items() if any(x is None for x in v)})
        results.append({'path':s['path'],'rows':s['metadata']['3'],'first_row':row,'pv_counts':counts,'waveform_lengths':lengths,'event_id_unique':len(set(cols['event_id'])),'cycle_id_unique':len(set(cols['cycle_id']))})
    Path('data/dd_inspected/sample_rows.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
