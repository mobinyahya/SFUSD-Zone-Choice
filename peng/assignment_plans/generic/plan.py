'''
Created on Jun 19, 2013

@author: pengshi
'''
import abc
from collections.abc import Container, Iterable, Sized
from collections import OrderedDict
from peng.utils import data_saver, data_reader
from peng.utils.data_equal import DataEqualable
from peng.utils.common import _getArg

class Plan(Container, Iterable, Sized, DataEqualable, metaclass=abc.ABCMeta):
    '''An abstract class that represents a certain assignment plan'''
    @abc.abstractproperty
    def iset(self):
        pass
    
    @abc.abstractmethod
    def choiceSet(self,i):
        pass
    
    @abc.abstractmethod
    def weight(self,i):
        pass
    
    @abc.abstractproperty
    def name(self):
        pass
    
    def __contains__(self,i):
        return i in self.iset
    
    def __len__(self):
        return len(self.iset)
    
    def __iter__(self):
        return iter(self.iset)
    
    def totalJSet(self):
        ans=set()
        for i in self.iset:
            ans=ans.union(self.choiceSet(i))
        return ans
    
            
    
    def averageNumChoices(self):
        return self.totalNumChoices()/float(self.totalWeights())
    
    def totalWeights(self):
        return sum(self.weight(i) for i in self)
        
    def totalNumChoices(self):
        return sum(len(self.choiceSet(i))*self.weight(i) for i in self)
    
    def totalSymmetricDiff(self,plan2):
        if not isinstance(plan2,Plan):
            raise TypeError
        ans=0
        for i in self:
            if not i in plan2:
                ans+=len(self.choiceSet(i))*self.weight(i)
            else:
                ans+=len(self.choiceSet(i).symmetric_difference(plan2.choiceSet(i)))*self.weight(i)
        return ans
    
    def totalNotHavePct(self,plan2):
        if not isinstance(plan2,Plan):
            raise TypeError
        ans=0
        for i in self:
            ans+=len(plan2.choiceSet(i)-self.choiceSet(i))*self.weight(i)
        return ans/float(self.totalNumChoices())
    
    def totalAdditionalPct(self,plan2):
        if not isinstance(plan2,Plan):
            raise TypeError
        ans=0
        for i in self:
            if not i in plan2:
                ans+=len(self.choiceSet(i))*self.weight(i)
            else:
                ans+=len(self.choiceSet(i)-plan2.choiceSet(i))*self.weight(i)
        return ans/float(self.totalNumChoices())
    
    def diffPct(self,plan2):
        return self.totalSymmetricDiff(plan2)/float(self.totalNumChoices())
        
    def listRep(self):
        ans=OrderedDict()
        for i in self:
            ans[i]=list(self.choiceSet(i))
        return ans
        
    def saveChoiceSets(self,filename):
        data_saver.saveLists(filename, self.listRep(), 1)
        
    def accessSet(self,j):
        ans=set()
        for i in self:
            if j in self.choiceSet(i):
                ans.add(i)
        return ans
    
    def dataEquals(self,obj):
        if not isinstance(obj,Plan):
            return False
        if not self.iset==obj.iset:
            return False
        for i in self:
            if not self.weight(i)==obj.weight(i):
                return False
            if not self.choiceSet(i)==obj.choiceSet(i):
                return False
        return True
    
    def getChoiceSetDict(self):
        ans={i:self.choiceSet(i) for i in self.iset}
        return ans
        
class SimplePlan(Plan):
    def __init__(self, choiceSetDict, weightDict, **kwargs):
        self._choiceSetDict=OrderedDict()
        self._weightDict=OrderedDict()
        self._name=_getArg(kwargs,'name','SimplePlan')
        for i in weightDict:
            if i in choiceSetDict:
                self._choiceSetDict[i]=set(choiceSetDict[i])
            else:
                self._choiceSetDict[i]=set()
            self._weightDict[i]=weightDict[i]
            
        self._iset=list(self._choiceSetDict.keys())
       
    @staticmethod
    def fromFile(choiceSetFile,weightFile,**kwargs):
        choiceSetList=data_reader.getLists(choiceSetFile, 1)
        choiceSetDict=OrderedDict()
        for i in choiceSetList:
            choiceSetDict[i]=set(choiceSetList[i])
        weightDict=data_reader.getDict(weightFile, 1, True)
        return SimplePlan(choiceSetDict,weightDict,**kwargs)
       
    @property 
    def iset(self):
        return self._iset
    
    def choiceSet(self,i):
        return self._choiceSetDict[i]
    
    def weight(self,i):
        return self._weightDict[i]    
    
    @property
    def name(self):
        return self._name
    