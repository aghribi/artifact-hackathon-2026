"""Re-evaluate both backend orbit matrices and weighted SVDs at all 144 nodes (L4H drift)."""
import argparse,json,time,hashlib,shutil
from pathlib import Path
from multiprocessing import Pool
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'orbit_matching/results/vbc_on_l4h_off/orbit'


def work(backend):
    import torch
    torch.set_num_threads(1)
    from .vbc_reference import ReferenceOrbitEngine
    engine=ReferenceOrbitEngine(backend)
    with np.load(ROOT/'optics_matching_vbc_on/results/rematched_grid_vbc_on.npz') as d:axis=d['spot_sizes_m']
    start=time.time()
    for i,x in enumerate(axis):
        for j,y in enumerate(axis):
            path=OUT/'checkpoints'/f'{backend}_{i:02d}_{j:02d}.npz'
            if path.exists():continue
            result=engine.evaluate(float(x),float(y));record={}
            for plane in 'xy':
                d=result[plane];a=d['weighted_response'];u,s,v=np.linalg.svd(a,full_matrices=False)
                for mode in range(len(s)):
                    if v[mode,np.argmax(abs(v[mode]))]<0:v[mode]*=-1;u[:,mode]*=-1
                for key,value in d.items():record[f'{plane}_{key}']=value
                record.update({f'{plane}_U':u,f'{plane}_singular_values':s,f'{plane}_Vt':v,
                    f'{plane}_rank':int(np.sum(s>s[0]*1e-8))})
            np.savez_compressed(path,**record)
        print(f'{backend} orbit row {i+1}/12; {time.time()-start:.1f}s',flush=True)
    return backend


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=2);parser.add_argument('--rebuild',action='store_true')
    args=parser.parse_args();(OUT/'checkpoints').mkdir(parents=True,exist_ok=True)
    from optics_matching_vbc_on import MatchedOptics
    optics=MatchedOptics(l4h_off=True);identity=OUT/'inputs.json'
    current=dict(optics_fingerprint=optics.configuration_fingerprint(),reference_engine_sha256=hashlib.sha256((ROOT/'orbit_matching/vbc_reference.py').read_bytes()).hexdigest(),L4H='drift',weights=dict(bpm=1,ip=5),step_rad=1e-7)
    changed=identity.exists() and json.loads(identity.read_text())!=current
    if changed and not args.rebuild:raise RuntimeError('Orbit inputs changed. Run again with --rebuild to archive and re-evaluate.')
    if args.rebuild and identity.exists():
        archive=OUT/'archive'/str(time.time_ns());archive.mkdir(parents=True)
        shutil.move(str(OUT/'checkpoints'),str(archive/'checkpoints'))
        shutil.move(str(identity),str(archive/'inputs.json'))
        (OUT/'checkpoints').mkdir()
    identity.write_text(json.dumps(current,indent=2)+'\n')
    with Pool(args.workers) as pool:list(pool.imap_unordered(work,['cheetah','xsuite']))
    from .compile_vbc_orbits import compile_results
    compile_results()

if __name__=='__main__':main()
