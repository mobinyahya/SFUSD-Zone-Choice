'''
Created on Dec 27, 2012

@author: pengshi
'''
import collections
from peng.utils.data_saver import PickleSaveLoad
from peng.utils.data_equal import DataEqualable
from peng.utils import data_equal, data_saver
import numpy as np
from peng.deferred_acceptance.chooser import Chooser
from peng.deferred_acceptance.bin import Bin

class Assignment(collections.MutableMapping,PickleSaveLoad,DataEqualable):
    ''' A helper class to encode a certain mapping. It is used by the deferred acceptance algorithm
    to encode an assignment of agents to items.'''
    def __init__(self,initDict={},none=None):
        self._map={}
        self._assignedTo={}
        self._noneSet=set()
        self._none=none
        for k in initDict:
            self[k]=initDict[k]
    
    def __len__(self):
        return len(self._map)
    
    def __iter__(self):
        return iter(self._map)
    
    def __contains__(self,key):
        return key in self._map
    
    def __getitem__(self,key):
        if key not in self:
            return None
        return self._map[key]
    
    def __setitem__(self,key,value):
        #remove from current reverseMap
        if key in self._map:
            del self[key]
            
        if value==self._none:
            self._map[key]=None
            self._noneSet.add(key)
        else:
            self._map[key]=value
            if not value in self._assignedTo:
                self._assignedTo[value]=set()
            self._assignedTo[value].add(key)
            
    def __delitem__(self,key):
        if not key in self:
            raise ValueError
        value=self[key]
        if value==None:
            self._noneSet.discard(key)
        else:
            self._assignedTo[value].discard(key)
            if not len(self._assignedTo[value]):
                del self._assignedTo[value]
        del self._map[key]
        
    @property
    def dict(self):
        return self._map
        
    def assignedTo(self,value):
        if value==None:
            return self._noneSet
        elif value in self._assignedTo:
            return self._assignedTo[value]
        else:
            return set()
        
    def values(self):
        ans=list(self._assignedTo.keys())
        if len(self._noneSet):
            ans+=[None]
        return ans
     
    def __str__(self):
        rows=['Assignment with %d entries'%len(self._map)]
        for v in list(self.values()):
            rows.append('\t %s: %s'%(str(v),','.join([str(k) for k in self.assignedTo(v)])))
        return '\n'.join(rows)
    
    def dataEquals(self,obj):
        if not isinstance(obj,Assignment):
            return False
        return data_equal.equal(self.dict,obj.dict)
            
    def subsetKeys(self,keys):
        ans=Assignment()
        for k in keys:
            ans[k]=self[k]
        return ans
            
    @staticmethod
    def fromChooserBinAssignment(a,collapseChooser=True,collapseBin=True):
        ans=Assignment()
        for c in a:
            if collapseChooser and isinstance(c,Chooser):
                s=c.identity
            else:
                s=c
            if a[c]==None:
                p=None
            else:
                b=a[c]
                if collapseBin and isinstance(b,Bin):
                    p=b.program
                else:
                    p=b
            ans[s]=p
        return ans
    
    def matchPct(self,assignment):
        diff=len(self.diffKeys(assignment))
        denum=len(self)
        if denum:
            return (denum-diff)/float(denum)
        else:
            return 1
    
    def diffKeys(self,assignment):
        ans=[]
        for k in self:
            if not k in assignment or self[k]!=assignment[k]:
                ans.append(k)
        return ans

    def graphMatch(self,ax,assignment,vlabels={},**kwargs):
        values=list(self.values())
        if not vlabels:
            vlabels={v:str(v) for v in values}
        vdiff=[self.assignedToMatchPct(v,assignment) for v in values]
        pos=np.arange(len(values))
        width=0.4
        rects=ax.bar(pos,vdiff,width,**kwargs)
        ax.set_ylabel('% Match (% of assignment also assigned in other)')
        ax.set_xlabel('Assigned to Value')
        label=kwargs['label'] if 'label' in kwargs else True
        if label:
            ax.set_xticks(pos+width/2)
            ax.set_xticklabels([vlabels[v] for v in values])
        return ax
    
    def assignedToMatchPct(self,v,assignment):
        ownSet=self.assignedTo(v)
        if len(ownSet):
            denum=float(len(ownSet))
            num=0
            otherSet=assignment.assignedTo(v)
            for k in ownSet:
                if k in otherSet:
                    num+=1
            return num/denum
        else:
            return 1
        
    def outputCSV(self,filename,collapseChooser=False):
        if collapseChooser:
            ans={s.id: self[s] for s in self}
        else:
            ans=self
        data_saver.saveDict(filename, ans, 1)
        
    def outputMatchCSV(self,filename,assignment):
        values=list(self.values())
        ans={}
        for v in values:
            ans[v]=self.assignedToMatchPct(v, assignment)
        data_saver.saveDict(filename, ans,1)
            