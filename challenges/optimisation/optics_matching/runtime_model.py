"""Portable differentiable energy-resolved model of the four final quadrupoles."""
from pathlib import Path
import numpy as np
import torch
DEFAULT=Path(__file__).resolve().parent/'data'/'beamlet_runtime.npz'

class RuntimeFinal:
 def __init__(self,path=DEFAULT):
  with np.load(path) as d:
   self.w=torch.tensor(d['weights'],dtype=torch.float64);self.mu=torch.tensor(d['means'][:,:4],dtype=torch.float64);self.cov=torch.tensor(d['covariances'][:,:4,:4],dtype=torch.float64)
   self.delta=torch.tensor(d['delta_before'][0],dtype=torch.float64);self.L=d['quad_lengths'];self.drifts=torch.tensor(d['fixed_maps'][:,:,0,1],dtype=torch.float64)
  self.stations=9;self.radii=torch.tensor([.020,.020,.068,.068],dtype=torch.float64)
 def quad(self,k,L):
  d=1+self.delta
  if torch.as_tensor(L).ndim:L=L[...,None];d=d[None,:]
  kp=k/d;kp,L=torch.broadcast_tensors(kp,torch.as_tensor(L,dtype=torch.float64));d=torch.zeros_like(kp)+d
  results=[]
  for strength,sign in [(kp,-1),(-kp,1)]:
   qp=torch.sqrt(torch.clamp(strength,min=1e-8));qm=torch.sqrt(torch.clamp(-strength,min=1e-8))
   C=torch.where(strength>=0,torch.cos(qp*L),torch.cosh(qm*L));S=torch.where(strength>=0,torch.sin(qp*L)/qp,torch.sinh(qm*L)/qm)
   C=torch.where(strength.abs()<1e-6,1-strength*L**2/2+strength**2*L**4/24,C);S=torch.where(strength.abs()<1e-6,L-strength*L**3/6+strength**2*L**5/120,S)
   results.append(torch.stack([torch.stack([C,S/d],-1),torch.stack([sign*k*S,C],-1)],-2))
  return results
 def moments(self,X,Y):
  z=torch.zeros_like(X[...,0,0]);hx=torch.stack([X[...,0,0],X[...,0,1],z,z],-1);hy=torch.stack([z,z,Y[...,0,0],Y[...,0,1]],-1);H=torch.stack([hx,hy],-2)
  means=(H@self.mu[...,None]).squeeze(-1);cov=H@self.cov@H.transpose(-1,-2)
  mu=torch.sum(means*self.w[...,None],dim=-2);raw=torch.sum((cov+means[...,None]*means[...,None,:])*self.w[...,None,None],dim=-3)
  return mu,raw-mu[...,None]*mu[...,None,:]
 def evaluate(self,k,apertures=False):
  X=torch.eye(2,dtype=torch.float64).expand(len(self.w),2,2);Y=X;ratios=[]
  for j in range(5):
   z=torch.zeros_like(self.drifts[j]);one=torch.ones_like(z);D=torch.stack([torch.stack([one,self.drifts[j]],-1),torch.stack([z,one],-1)],-2);X=D@X;Y=D@Y
   if j<4:
    if apertures:
     A,B=self.quad(k[j],self.L[j]*torch.linspace(0,1,self.stations,dtype=torch.float64));mu,cov=self.moments(A@X,B@Y)
     largest=(cov[...,0,0]+cov[...,1,1]+torch.sqrt((cov[...,0,0]-cov[...,1,1])**2+4*cov[...,0,1]**2+1e-36))/2
     ratios.append((torch.sqrt(mu.square().sum(-1)+1e-36)+3*torch.sqrt(largest))/self.radii[j])
    A,B=self.quad(k[j],self.L[j]);X=A@X;Y=B@Y
  _,cov=self.moments(X,Y);sizes=torch.sqrt(torch.stack([cov[0,0],cov[1,1]]))
  return (sizes,torch.cat(ratios)) if apertures else sizes
 def correct(self,initial,log_targets,steps=6,derivatives=False):
  # Fixed-count analytic Gauss-Newton updates preserve smoothness of the surface.
  k=initial
  if not k.requires_grad:k=k.clone().requires_grad_(True)
  for _ in range(steps):
   log_sizes=torch.log(self.evaluate(k))
   J=torch.stack([torch.autograd.grad(log_sizes[p],k,retain_graph=True,create_graph=derivatives)[0] for p in range(2)])
   step=J.T@torch.linalg.solve(J@J.T+1e-10*torch.eye(2,dtype=torch.float64),log_sizes-log_targets)
   step=step/torch.sqrt(1+step.square().sum()/4)
   k=k-step
   if not derivatives:k=k.detach().requires_grad_(True)
  return k
