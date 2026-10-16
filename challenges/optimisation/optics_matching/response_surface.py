"""C2 optics response surface induced by the matched final-focus strength family."""
from pathlib import Path
import numpy as np
DEFAULT=Path(__file__).resolve().parent/'data'/'optics_response_model.npz'

def functions(k,L):
 if abs(k)<1e-8:
  return 1-k*L*L/2+k*k*L**4/24,L-k*L**3/6+k*k*L**5/120
 q=np.lib.scimath.sqrt(k)
 return np.cos(q*L),np.sin(q*L)/q

def quad(k,L):
 c,s=functions(k,L);d,t=functions(-k,L)
 return np.array([[c,s,0,0],[-k*s,c,0,0],[0,0,d,t],[0,0,k*t,d]],dtype=np.result_type(k,complex))

class OpticsResponse:
 def __init__(self,path=DEFAULT):
  with np.load(path) as d:self.data={key:d[key] for key in d.files}
 def tail_matrix(self,k):
  T=np.eye(4,dtype=complex)
  for j,L in enumerate(self.data['drift_lengths']):
   D=np.eye(4,dtype=complex);D[0,1]=L;D[2,3]=L;T=D@T
   if j<4:T=quad(k[j],self.data['quad_lengths'][j])@T
  return T
 def evaluate(self,full_k1):
  k=np.asarray(full_k1,float)
  if k.shape!=(46,) or not np.allclose(k[:42],self.data['common_k1'][:42],rtol=0,atol=1e-10):
   raise ValueError('This response model requires the saved common first 42 quadrupole strengths')
  P=self.data['prefix'];T=np.real(self.tail_matrix(k[-4:]));M=T@P
  derivatives=np.empty((46,4,4));derivatives[:42]=np.einsum('ij,njk->nik',T,self.data['prefix_derivatives'])
  for j in range(4):
   z=k[-4:].astype(complex);z[j]+=1e-20j
   derivatives[42+j]=self.tail_matrix(z).imag/1e-20@P
  response=self.data['constant_response'].copy()
  beta=[]
  for plane,o in enumerate([0,2]):
   a,b,c,d=M[o,o],M[o,o+1],M[o+1,o],M[o+1,o+1];B=self.data['initial_beta'][plane];A=self.data['initial_alpha'][plane];G=(1+A*A)/B;N=B*a*a-2*A*a*b+G*b*b;det=a*d-b*c
   gradient=np.array([[(2*B*a-2*A*b)/N-d/det,(-2*A*a+2*G*b)/N+c/det],[b/det,-a/det]])
   response[-2+plane]=np.einsum('ij,nij->n',gradient,derivatives[:,o:o+2,o:o+2]);beta.append(N/det)
  return dict(response=response,weighted_response=np.sqrt(self.data['row_weights'])[:,None]*response,ip_beta_m=np.array(beta),row_weights=self.data['row_weights'].copy(),quadrupoles=self.data['quadrupoles'].tolist(),screens=self.data['screens'].tolist())
