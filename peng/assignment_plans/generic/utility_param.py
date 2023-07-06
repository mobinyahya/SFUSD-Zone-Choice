'''
Created on Jun 27, 2013

@author: pengshi
'''
import abc
from peng.utils.common import _getArg
import numpy as np
from peng.utils.data_equal import DataEqualable
from peng.utils import data_equal
from collections import Counter

class UtilityParam(DataEqualable, metaclass=abc.ABCMeta):
    '''An abstract class that implements a MNL model'''
    @abc.abstractproperty
    def iset(self):
        pass
    
    @abc.abstractmethod
    def jset(self,i):
        pass
    
    @abc.abstractmethod
    def baseUtility(self,i,j):
        pass
    
    @abc.abstractmethod
    def idioSize(self,i):
        pass

    @property
    def euler(self):
        return 0.577215664901532
    
    def utilityOffset(self,i):
        return 0.577215664901532*self.idioSize(i)
    
    def totalJSet(self):
        ans=set()
        for i in self.iset:
            ans=ans.union(self.jset(i))
        return ans
    
    def menuUtility(self,i,M):
        inner=self._expSum(i,M)
        if inner>0:
            return np.log(inner)*self.idioSize(i)
        else:
            return -np.Inf 
    
    def _expSum(self,i,M):
        idioSize=self.idioSize(i)
        return sum(np.exp(self.baseUtility(int(i), int(j))/idioSize)for j in M)
    
    def attractionWeight(self,i,j):
        return np.exp(self.baseUtility(i, j)/self.idioSize(i))
    
    def demandProb(self,i,M):
        tot=self._expSum(i,M)
        ans=Counter()
        if tot>0:
            idioSize=self.idioSize(i)
            for j in M:
                ans[j]=np.exp(self.baseUtility(i, j)/idioSize)/tot
        return ans
    
    def randomUtility(self,i,j):
        return self.baseUtility(i, j)+np.random.gumbel(0,self.idioSize(i))
    
    def randomChoices(self,i,M):
        tmp=sorted([(self.randomUtility(i, j),j) for j in M],reverse=True)
        ans=[]
        for k in range(len(tmp)):
            if tmp[k][1]==None:
                break
            ans.append(tmp[k][1])
        return ans
        
    def dataEquals(self,obj):
        if not isinstance(obj,UtilityParam):
            return False
        if not data_equal.equal(self.iset, obj.iset):
            return False
        for i in self.iset:
            if self.idioSize(i)!=obj.idioSize(i):
                return False
            if not data_equal.equal(self.jset(i), obj.jset(i)):
                return False
            for j in self.jset(i):
                if self.baseUtility(i, j)!=obj.baseUtility(i,j):
                    return False
        return True
    
class SimpleUtilityParam(UtilityParam):
    def __init__(self,baseUtilityDict,**kwargs):
        self._baseUtility={}

        self._idioSize=float(_getArg(kwargs,'idioSize',1))
        for i in baseUtilityDict:
            self._baseUtility[i]={}
            for j in baseUtilityDict[i]:
                self._baseUtility[i][j]=baseUtilityDict[i][j]
                                
        self._iset=set()
        self._jset={}
        for i in baseUtilityDict:
            self._iset.add(i)
            self._jset[i]=set()
            for j in baseUtilityDict[i]:
                self._jset[i].add(j)
                
    @property
    def iset(self):
        return self._iset
    
    
    def jset(self,i):
        return self._jset[i]
    
    def baseUtility(self,i,j):
        if not i in self._baseUtility or j not in self._baseUtility[i]:
            return -np.Inf
        else:
            return self._baseUtility[i][j]
        
    
    def idioSize(self,i):
        return self._idioSize