'''
Created on Dec 27, 2012

@author: pengshi
'''
import abc
from peng.deferred_acceptance.chooser import Chooser
from peng.deferred_acceptance.bin import Bin
from peng.deferred_acceptance.assignment import Assignment
import heapq

class DAA(object, metaclass=abc.ABCMeta):
    ''' A generic implementation of the Deferred Acceptance (DA) Algorithm. '''
    def __init__(self,choosers,bins):
        for c in choosers:
            if not isinstance(c, Chooser):
                raise TypeError
        for b in bins:
            if not isinstance(b,Bin):
                raise TypeError
        self._choosers=choosers
        self._bins=bins
        self._binSet=set(bins)
        self._init()
        
    @property
    def choosers(self):
        return self._choosers
    
    @property
    def bins(self):
        return self._bins
    
    @property
    def assignment(self):
        return self._assignment
        
    @abc.abstractmethod
    def score(self,bin,chooser):
        pass
    
    def getCutoff(self,bin):
        myQ=self._heap[bin]
        if len(myQ)<bin.capacity:
            return -1e7
        elif len(myQ)==0:
            return None
        else:
            return myQ[0][0]
    
    def _init(self):
        self._assignment=Assignment()
        self._choiceIndex={c:0 for c in self._choosers}
        self._heap={b:[] for b in self._bins}
        for h in self._heap.values():
            heapq.heapify(h)
    
    def assign(self):
        self._init()
        for c in self._choosers:
            if not c in self._assignment:
                self._assignment[c]=None
                self._tryAssign(c)
        return self._assignment
    
    def numUnassigned(self):
        return sum(1 for c in self._assignment if self._assignment[c] is None)
    
    def _curChoice(self,c):
        if (self._choiceIndex[c]>=len(c.choices)):
            return None
        b=c.choices[self._choiceIndex[c]]
        if b in self._binSet:
            return b
        else:
            self._choiceIndex[c]+=1
            return self._curChoice(c)
        
    
    def _tryAssign(self,c):
        assert(self._assignment[c]==None) #invariant
        #get next possible choice
        b=self._curChoice(c)
        if b==None:
            return        
        
        heapitem=(self.score(b,c),c)
        
        if len(self._heap[b])<b.capacity:
            #try to assign if capacity remains
            self._assignment[c]=b
            heapq.heappush(self._heap[b], heapitem)
        else:
            #if capacity full, bump out lowest score guy (might be self), increase ind of that guy by 1, recurse on that guy
            c2=heapq.heappushpop(self._heap[b], heapitem)[1]
            self._choiceIndex[c2]+=1
            if c!=c2:
                self._assignment[c]=b
                self._assignment[c2]=None
            self._tryAssign(c2)
            
class SimpleDAA(DAA):
    def __init__(self,choosers,bins):
        for b in bins:
            if not hasattr(b, 'score') or type(b.score)!=dict:
                raise TypeError
        super(SimpleDAA,self).__init__(choosers,bins)
        
    def score(self,bin,chooser):
        return bin.score[chooser.identity]
    
class DASTB(DAA):
    def __init__(self,choosers,bins,lotteries):
        super(DASTB,self).__init__(choosers,bins)
        self._lotteries=lotteries
        
    def score(self,bin,chooser):
        return self._lotteries[chooser.identity]+bin.scoreBoost(chooser.identity)
        