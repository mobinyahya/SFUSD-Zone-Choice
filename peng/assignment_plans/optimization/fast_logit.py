'''
Created on Jul 24, 2013

@author: pengshi
'''
import numpy as np
from peng.utils.common import _getArg


class FastLogit(object):
    def __init__(self,u,beta,**kwargs):
        ''' A helper class that implements various operations related with the MNL model.
        It uses numpy array operations if possible for efficiency.'''
        self._T,self._S=u.shape
        self._beta=np.array(beta,dtype=float)
        self._u=u
        self._exp=np.exp(np.einsum('ij,i->ij', u, np.reciprocal(self._beta)))
        if 'iind' in kwargs:
            self._iind=kwargs['iind']
            self._iset=set(self._iind.keys())
        else:
            self._iset=set(range(self._T))
            self._iind={i:i for i in range(self._T)}

        if 'jind' in kwargs:
            self._jind=kwargs['jind']
            self._jset=set(self._jind.keys())
        else:
            self._jset=set(range(self._S))
            self._jind={j:j for j in range(self._S)}

    def sample(self,t):
        return self._u[t]+np.random.gumbel(0,self._beta[t],self._S)

    def iind(self,i):
        return self._iind[i]

    def jind(self,j):
        return self._jind[j]

    @property
    def beta(self):
        return self._beta

    @property
    def iset(self):
        return self._iset

    @property
    def jset(self):
        return self._jset

    @staticmethod
    def fromUtilParam(utilParam):
        iset=list(utilParam.iset)
        jset=list(utilParam.totalJSet())
        T=len(iset)
        S=len(jset)
        iind={iset[t]:t for t in range(len(iset))}
        jind={jset[s]:s for s in range(len(jset))}
        u=np.ones((T,S))*(-np.Inf)
        beta=np.ones(T)

        for i in iset:
            t=iind[i]
            beta[t]=utilParam.idioSize(i)

            for j in utilParam.jset(i):
                s=jind[j]
                u[t,s]=utilParam.baseUtility(i,j)
        return FastLogit(u,beta,iind=iind,jind=jind)

    @property
    def T(self):
        return self._T

    @property
    def S(self):
        return self._S

    def w(self, t):
        return self._exp[t]

    def menuUtility(self, i, Menu):
        return self.vp(self.iind(i),[self.jind(j) for j in Menu], vOnly=True)

    def vp(self, t, M, vOnly=False, pOnly=False):
        if not len(M):
            v = -np.inf
            p = np.array([])
        else:
            num = self._exp[t, M]
            sumexp = np.sum(num)
            v = np.log(sumexp)*self._beta[t]
            p = num/sumexp

        if vOnly:
            return v
        elif pOnly:
            return p
        else:
            return v, p
