"""Numerical interpolation and local sign-transport helpers."""

import numpy as np

from scipy.interpolate import make_interp_spline

from scipy.sparse import coo_matrix

from scipy.sparse.csgraph import minimum_spanning_tree

def compile_surface(axis, values):
    first=make_interp_spline(axis,values,k=3,axis=0)
    second=make_interp_spline(axis,first.c,k=3,axis=1)
    return first.t,second.t,np.moveaxis(second.c,0,1)

def align_modes(vectors):
    """Maximum-overlap spanning-tree gauge; retain sorted mode indices."""
    n=len(vectors);rank=vectors.shape[2];v=vectors.copy()
    node=np.arange(n*n).reshape(n,n)
    a=np.r_[node[:-1].ravel(),node[:,:-1].ravel()]
    b=np.r_[node[1:].ravel(),node[:,1:].ravel()]
    original=v.reshape(n*n,rank,v.shape[-1])
    for mode in range(rank):
        overlap=np.einsum('ij,ij->i',original[a,mode],original[b,mode])
        graph=coo_matrix((-np.maximum(abs(overlap),1e-15),(a,b)),shape=(n*n,n*n)).tocsr()
        tree=minimum_spanning_tree(graph)
        tree=(tree+tree.T).tocsr()
        signs=np.zeros(n*n);root=(n//2)*n+n//2
        row=original[root,mode];signs[root]=1 if row[np.argmax(abs(row))]>=0 else -1
        stack=[root]
        while stack:
            i=stack.pop()
            for j in tree.indices[tree.indptr[i]:tree.indptr[i+1]]:
                if signs[j]==0:
                    signs[j]=signs[i]*(1 if original[i,mode]@original[j,mode]>=0 else -1)
                    stack.append(j)
        original[:,mode]*=signs[:,None]
    return v

def transport_path(self, target_sizes_m, n_modes=None, *, minimum_overlap=.98, max_depth=12, max_log_step=.05):
        """Continuously transport signs along a specified path in log-size space.

        Adaptive midpoint checks limit rotations, including through cells where
        a global spline sign convention has a seam. Actual degeneracies cannot
        be removed by sign alignment; unresolved segments are reported.
        Returns modes at requested points, plus internal-check diagnostics.
        """
        points = np.asarray(target_sizes_m,dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points)<1:
            raise ValueError('target_sizes_m must have shape (N,2), N>=1')
        if not 0 < minimum_overlap < 1 or max_depth < 0 or max_log_step <= 0:
            raise ValueError('Require 0 < minimum_overlap < 1 and max_depth >= 0')
        results=[self.evaluate(*points[0],n_modes)]
        stats=dict(internal_evaluations=0,unresolved_segments=0,minimum_accepted_overlap=1.)
        def advance(left_xy,left,right_xy,depth):
            right=self.evaluate(*right_xy,n_modes,reference_Vt=left['Vt'])
            stats['internal_evaluations']+=1
            overlap=float(np.min(np.einsum('ij,ij->i',left['Vt'],right['Vt'])))
            needs_subdivision = overlap<minimum_overlap or np.max(abs(np.log(right_xy/left_xy)))>max_log_step
            if needs_subdivision and depth<max_depth:
                middle_xy=np.exp((np.log(left_xy)+np.log(right_xy))/2)
                middle=advance(left_xy,left,middle_xy,depth+1)
                return advance(middle_xy,middle,right_xy,depth+1)
            if needs_subdivision:stats['unresolved_segments']+=1
            stats['minimum_accepted_overlap']=min(stats['minimum_accepted_overlap'],overlap)
            return right
        for i in range(1,len(points)):
            results.append(advance(points[i-1],results[-1],points[i],0))
        return dict(results=results,**stats)
