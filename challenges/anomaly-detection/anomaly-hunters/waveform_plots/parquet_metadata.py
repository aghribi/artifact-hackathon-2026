import io, struct, json, tarfile
from pathlib import Path

class Compact:
    def __init__(self, data): self.f=io.BytesIO(data)
    def byte(self): return self.f.read(1)[0]
    def varint(self):
        n=s=0
        while True:
            b=self.byte(); n|=(b&127)<<s
            if b<128: return n
            s+=7
    def value(self,t):
        if t in (1,2): return t==1
        if t==3: return self.byte()
        if t in (4,5,6):
            n=self.varint(); return (n>>1)^-(n&1)
        if t==7: return struct.unpack('<d',self.f.read(8))[0]
        if t==8: return self.f.read(self.varint()).decode('utf8',errors='replace')
        if t in (9,10):
            b=self.byte(); n=b>>4
            if n==15: n=self.varint()
            return [self.value(b&15) for _ in range(n)]
        if t==11:
            n=self.varint()
            if not n:return {}
            b=self.byte(); return {str(self.value(b>>4)):self.value(b&15) for _ in range(n)}
        if t==12: return self.obj()
        raise ValueError(t)
    def obj(self):
        result={}; last=0
        while True:
            b=self.byte()
            if not b:return result
            t=b&15; d=b>>4
            last=last+d if d else self.value(4)
            result[last]=self.value(t)

def footer(data):
    assert data[:4]==data[-4:]==b'PAR1'
    n=struct.unpack('<I',data[-8:-4])[0]
    return Compact(data[-8-n:-8]).obj()

if __name__=='__main__':
    base=Path.cwd(); output=base/'data/dd_inspected'; output.mkdir(exist_ok=True)
    summaries=[]
    for index,line in enumerate((base/'dd.list').read_text().splitlines(),1):
        path=Path(line.replace('/workspace/lustre/','/lustre/',1))
        with tarfile.open(path,'r:*') as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith('.parquet'):continue
                rel=Path(member.name)
                if rel.is_absolute() or '..' in rel.parts:raise ValueError(rel)
                dest=output/('%02d'%index)/rel
                source=archive.extractfile(member)
                source.seek(member.size-8)
                tail=source.read(8); assert tail[-4:]==b'PAR1'
                n=struct.unpack('<I',tail[:4])[0]
                source.seek(member.size-8-n)
                meta=Compact(source.read(n)).obj()
                if index in (2,3):
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    source.seek(0); dest.write_bytes(source.read())
                summaries.append({'archive':path.name,'member':member.name,'path':str(dest) if dest.exists() else None,'size':member.size,'metadata':meta})
    (output/'metadata.json').write_text(json.dumps(summaries,indent=2))
    print('files',len(summaries),'total rows',sum(s['metadata'][3] for s in summaries))
    schemas={json.dumps(s['metadata'][2],sort_keys=True) for s in summaries}
    print('distinct schemas',len(schemas))
    for schema in schemas: print('SCHEMA',schema)
    for s in summaries: print(s['archive'],s['member'],s['metadata'][3], 'rows')
    print('FIRST METADATA',json.dumps(summaries[0]['metadata'])[:18000])
