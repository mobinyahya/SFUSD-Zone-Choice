'''
Created on Feb 5, 2014

@author: pengshi
'''
from peng.deferred_acceptance.daa import DAA
import random
from peng.deferred_acceptance.bin import Bin
from peng.deferred_acceptance.chooser import Chooser
import numpy as np

class TypeDASTB(DAA):
    ''' Implementation of of DA with Single Tie-Breakers.'''
    def __init__(self,choosers,bins,typePrior):
        super(TypeDASTB,self).__init__(choosers,bins)
        self._lotteries={}
        for c in choosers:
            self._lotteries[c.identity]=random.random()
        self._typePrior=typePrior
        
    def score(self,bin,chooser):
        return self._lotteries[chooser.identity]+self._typePrior[chooser.type][bin.identity]
    
    def getAccess(self,bin,Type):
        cutoff=self.getCutoff(bin)
        if cutoff is None:
            return 0
        else:
            return min(max(1+self._typePrior[Type][bin.identity]-cutoff,0),1)
    
    
class SimpleBin(Bin):
    def __init__(self,identity,capacity):
        self._identity=identity
        self._capacity=capacity
        
    @property
    def identity(self):
        return self._identity
    
    @property
    def capacity(self):
        return self._capacity
    
    def __repr__(self):
        return f'Bin {self.identity} with capacity {self.capacity}'
    
class FastLogitChooser(Chooser):
    ''' A helper class for the simulating the DA algorithm under MNL utilities more efficient.'''
    def __init__(self,identity,myType,baseUtilities,menu,idioSize=1):
        k=len(baseUtilities)
        if len(menu)!=k:
            raise ValueError("len(menu)=%d not the same as len(baseltuiltiies)=%d"%(len(menu),k))
        self._utilities=baseUtilities+np.random.gumbel(0,idioSize,size=k)
        self._identity=identity
        self._type=myType
        self._menu=menu
        self._choices=menu[np.argsort(-self._utilities)]
        
    @property
    def identity(self):
        return self._identity
    
    @property
    def type(self):
        return self._type
    
    @property
    def choices(self):
        return self._choices
    
    def utility(self,sid):
        inds=np.where(self._menu==sid)[0]
        if len(inds)>0:
            return self._utilities[inds[0]]
        else:
            return -np.Inf
        
    def __str__(self):
        ans='FastLogitChooser %s: '%self.identity
        ans+=','.join(c.identity for c in self.choices)
        return ans