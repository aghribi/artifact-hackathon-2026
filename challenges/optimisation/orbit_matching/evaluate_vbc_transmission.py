"""Full-bunch transmission at all VBC-on rematched nodes, both backends."""
import argparse,json,os,time,hashlib
from pathlib import Path
from multiprocessing import Pool
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'orbit_matching/results/vbc_on'
GRID=ROOT/'optics_matching_vbc_on/results/rematched_grid_vbc_on.npz'
LATTICE=ROOT/'optics_matching_vbc_on/CLARA_cheetah_second_order.json'
STATE={}


def setup(backend,number,case):
    import torch
    torch.set_num_threads(1)
    from .vbc_machine import VBCMachine
    key=(backend,case)
    lattice=ROOT/'orbit_matching/results'/case/'lattice_l4h_off.json' if case=='vbc_on_l4h_off' else LATTICE
    if key not in STATE:STATE[key]=VBCMachine(backend,number,lattice_json=lattice)
    return STATE[key]


def task(spec):
    backend,i,number,case=spec;env=setup(backend,number,case)
    out=ROOT/"orbit_matching/results"/case
    with np.load(GRID) as d:k=d['k1'][i];names=d['quadrupoles'].tolist()
    rows=[]
    for j,values in enumerate(k):
        for name,value in zip(names,values):env.transport.write(name,'k1',float(value))
        env._simulate();r=env._readings
        rows.append(dict(j=j,transmission=float(r['transmission']),on_screen_fraction=float(r['on_screen_fraction']),
            stats=list(r['stats']),bpm_valid=r['bpm_valid'].tolist()))
    record=dict(backend=backend,i=i,num_particles=len(env._coordinates),rows=rows)
    (out/'transmission_checkpoints'/f'{backend}_{i:02d}.json').write_text(json.dumps(record)+'\n')
    return backend,i


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--particles',type=int,default=200000);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--l4h-off',action='store_true')
    args=parser.parse_args()
    case='vbc_on_l4h_off' if args.l4h_off else 'vbc_on'
    out=ROOT/'orbit_matching/results'/case
    out.mkdir(parents=True,exist_ok=True);(out/'transmission_checkpoints').mkdir(exist_ok=True)
    files=[GRID,LATTICE,ROOT/'nominal_model/xsuite_model.py',ROOT/'nominal_model/initial_distribution.npz',ROOT/'data/clara/lattice_data/apertures.json',ROOT/'RL_framework/rl_framework/envs/sim_env.py']
    if args.l4h_off:
        lattice=json.loads(LATTICE.read_text())
        length=lattice['elements']['cla_l4h_lin_cav_01'][1]['length']
        lattice['elements']['cla_l4h_lin_cav_01']=['Drift',dict(length=length,tracking_method='second_order')]
        path=out/'lattice_l4h_off.json';path.write_text(json.dumps(lattice,indent=2)+'\n');files.append(path)
    signature={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    signature['particles_requested']=args.particles
    identity=out/'transmission_inputs.json'
    if identity.exists() and json.loads(identity.read_text())!=signature:
        raise RuntimeError('Inputs changed; archive existing transmission checkpoints before rerunning')
    identity.write_text(json.dumps(signature,indent=2)+'\n')
    with np.load(GRID) as d:axis=d['spot_sizes_m'];angle=float(d['vbc_angle_rad'])
    tasks=[(b,i,args.particles,case) for b in ['cheetah','xsuite'] for i in range(len(axis)) if not (out/'transmission_checkpoints'/f'{b}_{i:02d}.json').exists()]
    start=time.time()
    with Pool(args.workers) as pool:
        for b,i in pool.imap_unordered(task,tasks):print(f'{b} row {i+1}/12 complete; elapsed {time.time()-start:.0f}s',flush=True)
    summary={}
    for backend in ['cheetah','xsuite']:
        transmission=np.empty((12,12));screen=np.empty((12,12));stats=np.empty((12,12,5));bpm=np.empty((12,12,31),dtype=bool)
        for i in range(12):
            record=json.loads((out/'transmission_checkpoints'/f'{backend}_{i:02d}.json').read_text())
            for r in record['rows']:
                j=r['j'];transmission[i,j]=r['transmission'];screen[i,j]=r['on_screen_fraction'];stats[i,j]=r['stats'];bpm[i,j]=r['bpm_valid']
        np.savez_compressed(out/f'transmission_{backend}.npz',spot_sizes_m=axis,transmission=transmission,
            on_screen_fraction=screen,screen_stats=stats,bpm_valid=bpm,num_particles=record['num_particles'],vbc_angle_rad=angle)
        summary[backend]=dict(min=float(transmission.min()),median=float(np.median(transmission)),max=float(transmission.max()),num_particles=record['num_particles'])
    (out/'transmission_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
