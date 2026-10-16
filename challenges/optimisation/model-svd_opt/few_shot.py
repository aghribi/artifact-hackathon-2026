"""Adaptive single-shot tuning. Model computations consume no measured shots."""
from pathlib import Path
from dataclasses import asdict
import json
import time
import numpy as np
import torch
import cheetah
from .controls import ModalSpace,prefix
from .models import Models
from optics_matching_vbc_on import MatchedOptics,lattice_document,build_cheetah
from nominal_model import xsuite_model as nominal


def size_error(info,target):
    if not info['on_screen']:return float('inf')
    return float(np.max(np.abs(np.array([info['sigma_x'],info['sigma_y']])/target-1)))


def centered(info,target,tolerance):
    return info['on_screen'] and np.max(np.abs(np.array([info['mu_x'],info['mu_y']])/target))<=tolerance


def loss(info,target,stage):
    if not info['on_screen']:return 1000.+100*(1-info['transmission'])
    residual=(np.array([info['sigma_x'],info['sigma_y']])/target-1) if stage=='optics' else np.array([info['mu_x'],info['mu_y']])/target
    return float(np.mean(np.abs(residual))+2*(1-info['on_screen_fraction']))


class SizeModel:
    """Nominal full-bunch transport, anchored to the latest measured size.

    Only nominal input particles and public setpoints enter this model. It never
    reads the episode beam, misalignments, RNG state or transport internals.
    """
    def __init__(self,models,particles):
        self.segment=build_cheetah(*models.target,model=models.optics,apertures=True,dtype=torch.float32)
        self.elements={e.name:e for e in self.segment.elements}
        coordinates,energy,_=nominal.load_distribution();coordinates=coordinates[:particles]
        self.particles=torch.tensor(np.c_[coordinates,np.ones(len(coordinates))],dtype=torch.float32)
        self.energy=torch.tensor(energy,dtype=torch.float32)
        self.extent=np.array([float(v) for v in self.elements[nominal.READOUT].extent])
    def predict(self,settings):
        for name,value in settings.items():
            if name in self.elements and hasattr(self.elements[name],'k1'):
                e=self.elements[name];e.k1.copy_(e.k1.new_tensor(value))
            elif name.endswith(('_h','_v')) and name[:-2] in self.elements:
                e=self.elements[name[:-2]];prop='horizontal_angle' if name.endswith('_h') else 'vertical_angle'
                if hasattr(e,prop):getattr(e,prop).copy_(getattr(e,prop).new_tensor(value))
        beam=cheetah.ParticleBeam(particles=self.particles.clone(),energy=self.energy)
        with torch.no_grad():beam=self.segment.track(beam)
        x=beam.x.numpy();y=beam.y.numpy();alive=beam.survival_probabilities.numpy()>.5
        on=alive&(abs(x)<=self.extent[1])&(abs(y)<=self.extent[3])
        if on.sum()<.1*len(on):return None
        return np.array([x[on].std(),y[on].std()])


class AdaptiveSession:
    """section-opt interface with one reusable seed reading and hard shot cap."""
    def __init__(self,machine,space,stage,target,config,budget,*,seed=True):
        self.machine=machine;self.space=space;self.stage=stage;self.target=np.array(target);self.config=config
        self.budget=budget;self.used=0;self.records=[];self.stopped=False;self.reason='shot_cap';self.stalled=0
        self.best_value=loss(machine.last[1],self.target,stage)
        self.seed_available=seed
    @property
    def remaining(self):return 0 if self.stopped else max(0,self.budget-self.used)+int(self.seed_available)
    def adequate(self,info):
        good=loss(info,self.target,'optics')<=self.config.focus_mismatch_tolerance if self.stage=='optics' else centered(info,self.target,self.config.center_tolerance)
        return good and info['on_screen_fraction']>=self.config.min_transmission
    def measure(self,u):
        from .workflow import BudgetExhausted
        if self.stopped or self.remaining<=0:raise BudgetExhausted()
        reused=self.seed_available and np.allclose(u,self.space.start,rtol=0,atol=1e-10)
        self.seed_available=False
        if reused:obs,info=self.machine.last
        else:
            if self.used>=self.budget:raise BudgetExhausted()
            obs,info=self.machine.shot(self.space.settings(u),self.stage);self.used+=1
        value=loss(info,self.target,self.stage);error=self.config.objective_noise_floor
        self.records.append(dict(u=np.array(u),value=value,error=error,obs=obs,info=info,reused_reading=bool(reused)))
        if not reused:
            improvement=self.best_value-value
            cost=.5*np.log2((51+info['evaluations'])/(50+info['evaluations']))
            self.stalled=0 if improvement>max(cost,error) else self.stalled+1
            self.best_value=min(self.best_value,value)
        if self.adequate(info):self.stopped=True;self.reason='target_tolerance'
        elif self.stalled>=self.config.patience:self.stopped=True;self.reason='no_significant_improvement'
        return value,error
    def should_stop_bo(self,optimizer,candidate):
        if not optimizer.models:return False
        from scipy.special import ndtr
        rng=np.random.default_rng(self.config.seed+len(self.records))
        points=np.vstack([candidate,rng.uniform(size=(512,len(candidate)))])
        mean,std=optimizer.models[-1].predict(optimizer.space.transform(points.tolist()),return_std=True)
        improvement=self.best_value-np.asarray(mean);std=np.maximum(std,1e-12);z=improvement/std
        ei=improvement*ndtr(z)+std*np.exp(-.5*z*z)/np.sqrt(2*np.pi)
        steps=self.machine.last[1]['evaluations'];cost=.5*np.log2((51+steps)/(50+steps))
        if float(ei.max())<cost:
            self.stopped=True;self.reason='expected_improvement_below_shot_cost';return True
        return False
    def best(self):return min(self.records,key=lambda r:r['value'])
    def search(self,method,seed):
        from .workflow import rcds_search,full_bo_search,BudgetExhausted
        if self.adequate(self.machine.last[1]):self.reason='existing_reading_adequate';return
        if not len(self.space.start) or self.budget<=0:self.reason='no_available_control_or_budget';return
        try:
            if method=='bo':full_bo_search(self,self.space.start,seed=seed,acquisition='ei')
            else:rcds_search(self.measure,self.space.start,self.remaining,seed=seed)
        except BudgetExhausted:pass
    def save(self,path):
        from .workflow import clean
        path.write_text(json.dumps(clean(dict(space=self.space.metadata(),records=self.records,shots=self.used,stop_reason=self.reason)),indent=2)+'\n')


def thread(machine,models,config,out,method):
    from .workflow import steering_space,rcds_search,BudgetExhausted,clean
    history=[];used=0
    while used<config.thread_budget:
        obs,info=machine.last
        if info['on_screen'] and np.all(obs['bpm_valid']):break
        before=prefix(obs['bpm_valid']);count=min(len(obs['bpm_valid']),before+1)
        space,R=steering_space(machine,models,count,config)
        mask=np.r_[obs['bpm_valid'][:count],obs['bpm_valid'][:count]].astype(bool)
        residual=np.r_[obs['bpm_x'][:count],obs['bpm_y'][:count]]
        if not len(space.start) or not mask.any():break
        u=space.candidate(R[mask],residual[mask]);settings=space.settings(u)
        if all(np.isclose(settings[n],machine.env.magnet_settings()[n],rtol=0,atol=1e-10) for n in settings):break
        machine.shot(settings,'thread');used+=1
        o,i=machine.last;history.append(dict(prefix_before=before,prefix_after=prefix(o['bpm_valid']),on_screen=i['on_screen']))
        if prefix(o['bpm_valid'])<=before and not i['on_screen']:
            # Restricted RCDS fallback when a linear correction cannot advance.
            remaining=min(3,config.thread_budget-used)
            if not remaining:break
            space,_=steering_space(machine,models,count,config)
            def measure(v):
                nonlocal used
                if used>=config.thread_budget:raise BudgetExhausted()
                ob,inf=machine.shot(space.settings(v),'thread');used+=1
                valid=ob['bpm_valid'][:count].astype(bool)
                value=100*np.count_nonzero(~valid)+float(np.mean(np.r_[ob['bpm_x'][:count][valid],ob['bpm_y'][:count][valid]]**2)/1e-6) if valid.any() else 1e4
                return value,config.objective_noise_floor
            rcds_search(measure,space.start,remaining,seed=config.seed)
    obs,info=machine.last
    result=dict(success=bool(info['on_screen'] and np.all(obs['bpm_valid'])),evaluations=used,progress=history,final=info)
    (out/f'{method}_threading.json').write_text(json.dumps(clean(result),indent=2)+'\n')
    return result


def optics(machine,models,beta,config,method,out,size_model):
    from .workflow import clean
    target=np.array(config.target);n=min(5,int(beta['rank']));s=beta['singular_values'][:n]
    space=ModalSpace(models.names,machine.env.magnet_settings(),beta['Vt'][:n].T,machine.env.magnet_limits,np.minimum(config.quad_cap,2/np.maximum(s,1e-12)))
    u=space.start.copy();warm=[];initial=machine.last;best=(loss(initial[1],target,'optics'),u.copy(),initial)
    used=0;reason='model_iteration_cap'
    for iteration in range(min(config.model_corrections,config.optics_budget)):
        info=machine.last[1]
        if loss(info,target,'optics')<=config.focus_mismatch_tolerance and info['on_screen_fraction']>=config.min_transmission:reason='target_tolerance';break
        baseline=size_model.predict({**machine.env.magnet_settings(),**space.settings(u)})
        if baseline is None or not len(u):reason='model_readout_unavailable';break
        J=np.zeros((2,len(u)))
        for j in range(len(u)):
            plus=u.copy();minus=u.copy();plus[j]=min(1.,u[j]+.025);minus[j]=max(0.,u[j]-.025)
            if plus[j]==minus[j]:continue
            a=size_model.predict({**machine.env.magnet_settings(),**space.settings(plus)})
            b=size_model.predict({**machine.env.magnet_settings(),**space.settings(minus)})
            if a is not None and b is not None:J[:,j]=(np.log(a)-np.log(b))/(plus[j]-minus[j])
        residual=np.log(np.array([info['sigma_x'],info['sigma_y']])/target)
        # Refine the candidate in the nominal model, updating its Jacobian
        # with a secant. These are model tracks, never environment shots.
        v=u.copy();r=residual.copy();best_model=(float(np.mean(abs(np.expm1(r)))),v.copy(),r.copy())
        lower=np.maximum(0.,u-config.model_trust);upper=np.minimum(1.,u+config.model_trust)
        for _ in range(5):
            delta=-J.T@np.linalg.solve(J@J.T+.01*np.eye(2),r)
            scale=1.
            for uj,dj,lo,hi in zip(v,delta,lower,upper):
                if dj>0:scale=min(scale,(hi-uj)/dj)
                elif dj<0:scale=min(scale,(lo-uj)/dj)
            change=max(0.,scale)*delta
            if np.linalg.norm(change)<1e-5:break
            proposal=np.clip(v+change,lower,upper)
            predicted_size=size_model.predict({**machine.env.magnet_settings(),**space.settings(proposal)})
            if predicted_size is None:break
            new_r=np.log(predicted_size/baseline)+residual
            J+=np.outer(new_r-r-J@change,change)/max(float(change@change),1e-12)
            v=proposal;r=new_r;value=float(np.mean(abs(np.expm1(r))))
            if value<best_model[0]:best_model=(value,v.copy(),r.copy())
            if value<.02:break
        candidate=best_model[1];predicted=best_model[2]
        gain=float(np.mean(np.abs(np.expm1(residual)))-best_model[0])
        cost=.5*np.log2((51+info['evaluations'])/(50+info['evaluations']))
        if gain<max(config.objective_noise_floor,cost):reason='predicted_gain_below_cost_or_noise';break
        obs,info=machine.shot(space.settings(candidate),'optics');used+=1;u=candidate
        value=loss(info,target,'optics');warm.append(dict(u=u.copy(),value=value,error=config.objective_noise_floor,obs=obs,info=info,reused_reading=False,proposal='anchored_model'))
        if value<best[0]:best=(value,u.copy(),machine.last)
        if not info['on_screen']:break
        if len(warm)>1 and warm[-2]['value']-value<config.objective_noise_floor:reason='model_stalled';break
    # Search in the original fixed five-mode subspace, with a local trust box.
    # Recenter the coordinate box at the latest measured point without spending
    # a shot on a duplicate baseline observation.
    widths=(space.high-space.low)*config.search_trust
    local=ModalSpace(space.names,machine.env.magnet_settings(),space.basis,machine.env.magnet_limits,widths)
    session=AdaptiveSession(machine,local,'optics',target,config,max(0,config.optics_budget-used))
    session.search(method,config.seed)
    # Restore a clearly better measured point only when needed, paying one shot.
    candidates=[(best[0],space.settings(best[1]))]
    if session.records:
        row=session.best();candidates.append((row['value'],local.settings(row['u'])))
    winner=min(candidates,key=lambda pair:pair[0]);current=loss(machine.last[1],target,'optics')
    if used+session.used<config.optics_budget and (not machine.last[1]['on_screen'] or current-winner[0]>2*config.objective_noise_floor):
        machine.shot(winner[1],'optics');used+=1
    payload=dict(space=space.metadata(),records=warm,search=dict(space=local.metadata(),records=session.records),shots=used+session.used,stop_reason=session.reason)
    (out/f'{method}_optics.json').write_text(json.dumps(clean(payload),indent=2)+'\n')
    return machine.last[1],payload['stop_reason']


def center(machine,models,config,method,out):
    from .workflow import clean
    target=np.array(config.target);settings=machine.env.magnet_settings();orbit=models.orbit_at(settings)
    models.save_orbits(orbit,f'{method}_final_optics_orbit');names=[];R=np.zeros((2,4))
    for i,p in enumerate('xy'):
        names.extend([n+('_h' if p=='x' else '_v') for n in orbit[p]['correctors'][-2:]])
        R[i,2*i:2*i+2]=orbit[p]['response'][-1,-2:]
    # Two resolved screen-position combinations, using exactly the final four
    # corrector channels; remove the two unobservable null directions.
    basis=R.T/np.maximum(np.linalg.norm(R,axis=1),1e-12)
    reach=4*np.maximum(np.abs(np.array([machine.last[1]['mu_x'],machine.last[1]['mu_y']])),target)/np.maximum(np.linalg.norm(R,axis=1),1e-12)
    space=ModalSpace(names,settings,basis,machine.env.magnet_limits,reach)
    records=[];used=0;reason='existing_reading_adequate';before_info=dict(machine.last[1]);before_position=np.array([machine.last[1]['mu_x'],machine.last[1]['mu_y']])
    if not centered(machine.last[1],target,config.center_tolerance) and config.center_budget:
        u=space.candidate(R,np.array([machine.last[1]['mu_x'],machine.last[1]['mu_y']]))
        obs,info=machine.shot(space.settings(u),'center');used+=1
        records.append(dict(u=u,value=loss(info,target,'center'),info=info,obs=obs,proposal='inverse_response'))
        reason='inverse_response'
    if records and machine.last[1]['on_screen'] and not centered(machine.last[1],target,config.center_tolerance) and used<config.center_budget:
        after=np.array([machine.last[1]['mu_x'],machine.last[1]['mu_y']])
        applied=np.array([machine.env.magnet_settings()[name]-settings[name] for name in names])
        prediction=R@applied;observed=after-before_position
        gain=np.ones(2);reliable=(abs(prediction)>.2*target)&(prediction*observed>0)
        gain[reliable]=np.clip(observed[reliable]/prediction[reliable],.25,4.)
        updated=gain[:,None]*R
        local=ModalSpace(names,machine.env.magnet_settings(),basis,machine.env.magnet_limits,reach)
        proposal=local.candidate(updated,after)
        ob,inf=machine.shot(local.settings(proposal),'center');used+=1;reason='measured_response_secant'
        records.append(dict(u=proposal,value=loss(inf,target,'center'),info=inf,obs=ob,proposal='measured_response_secant',response_gain=gain,space=local.metadata()))
    if not centered(machine.last[1],target,config.center_tolerance) and machine.last[1]['on_screen'] and used<config.center_budget:
        local=ModalSpace(names,machine.env.magnet_settings(),basis,machine.env.magnet_limits,target/np.maximum(np.linalg.norm(R,axis=1),1e-12))
        session=AdaptiveSession(machine,local,'center',target,config,max(0,config.center_budget-used-1))
        session.search('rcds',config.seed);used+=session.used;reason=session.reason
        # Every final readout remains a real single shot; no synthetic averaging.
        payload=dict(space=space.metadata(),records=records,search=dict(space=local.metadata(),records=session.records),shots=used,stop_reason=reason)
    else:payload=dict(space=space.metadata(),records=records,shots=used,stop_reason=reason)
    if not centered(machine.last[1],target,config.center_tolerance) and used<config.center_budget:
        candidates=[before_info]+[r['info'] for r in records]
        candidates += [r['info'] for r in payload.get('search',{}).get('records',[])]
        best=min(candidates,key=lambda i:loss(i,target,'center'))
        if not machine.last[1]['on_screen'] or loss(machine.last[1],target,'center')-loss(best,target,'center')>config.objective_noise_floor:
            machine.shot({n:best['magnets'][n] for n in names},'center');used+=1
            reason='restored_better_measured_setting';payload['shots']=used;payload['stop_reason']=reason
    (out/f'{method}_centering.json').write_text(json.dumps(clean(payload),indent=2)+'\n')
    return machine.last[1],reason


def run(config,out):
    from .workflow import Machine,RLConfig,SimEnv,clean
    started=time.perf_counter();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    if config.repeats!=1:raise ValueError('few_shot uses single-shot proposals; use --repeats 1 or --strategy fixed_budget')
    provider=MatchedOptics(l4h_off=True);provider.evaluate(*config.target)
    lattice=out/'episode_lattice.json';lattice.write_text(json.dumps(lattice_document(*config.target,model=provider))+'\n')
    (out/'config.json').write_text(json.dumps(clean(asdict(config)),indent=2)+'\n')
    models=Models(config.backend,config.target,out);models.save_orbits(models.orbit,'nominal_orbit');beta=models.beta_response()
    size_model=SizeModel(models,min(config.model_particles,config.particles));setup=time.perf_counter()-started
    cfg=RLConfig(backend=config.backend,nominal_lattice_json=str(lattice),num_particles=config.particles,target=(0.,0.,*config.target),
        max_episode_steps=config.thread_budget+config.optics_budget+config.center_budget+2,beam_variation=config.beam_variation,
        misalignment_variation=config.misalignment_variation,beam_jitter=config.beam_jitter,power_supply_ripple=config.ripple,bpm_noise=config.bpm_noise,
        include_cavity_phases=False,include_vbc_chicane=False)
    summary=dict(config=asdict(config),optics_fingerprint=provider.configuration_fingerprint(),branches={},pairing=dict(protocol='independent same-seed resets; no discovery or threading replay'),threading=dict(success=True),wall_time=dict(nominal_setup_and_calibration_s=setup))
    initial={}
    for method in ('bo','rcds'):
        branch_started=time.perf_counter();env=SimEnv(cfg);machine=Machine(env,out,1);machine.branch=method;obs,info=machine.reset(config.seed);initial[method]=clean(dict(obs={k:obs[k] for k in ('bpm_x','bpm_y','bpm_valid')},info=info))
        before=info;stage=time.perf_counter();threaded=thread(machine,models,config,out,method);thread_time=time.perf_counter()-stage
        stage=time.perf_counter()
        if threaded['success']:sized,optics_stop=optics(machine,models,beta,config,method,out,size_model)
        else:sized=machine.last[1];optics_stop='threading_failed';summary['threading']['success']=False
        optics_time=time.perf_counter()-stage;stage=time.perf_counter()
        if threaded['success'] and sized['on_screen']:final,center_stop=center(machine,models,config,method,out)
        else:final=sized;center_stop='screen_unavailable'
        status='completed' if threaded['success'] and final['on_screen'] else 'threading_failed' if not threaded['success'] else 'final_readout_lost'
        summary['branches'][method]=dict(status=status,before=before,after_optics=sized,final=final,threading=threaded,
            stop_reasons=dict(optics=optics_stop,centering=center_stop),target_met=bool(size_error(final,np.array(config.target))<=config.size_tolerance and centered(final,np.array(config.target),config.center_tolerance) and final['on_screen_fraction']>=config.min_transmission),
            wall_time=dict(threading_s=thread_time,optics_s=optics_time,centering_s=time.perf_counter()-stage,total_s=time.perf_counter()-branch_started))
        env.close();summary['wall_time']['total_s']=time.perf_counter()-started
        (out/'summary.json').write_text(json.dumps(clean(summary),indent=2)+'\n')
        print(f'{method}: {status}, shots={final["evaluations"]}, size_error={size_error(final,np.array(config.target)):.3f}, score={final["benchmark_score"]:.3f}',flush=True)
    summary['pairing']['initial_max_bpm_difference_m']=max(float(np.max(np.abs(np.array(initial['bo']['obs'][key])-initial['rcds']['obs'][key]))) for key in ('bpm_x','bpm_y'))
    (out/'summary.json').write_text(json.dumps(clean(summary),indent=2)+'\n');return summary
