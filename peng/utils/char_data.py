'''
Created on Jan 5, 2013

@author: pengshi
'''
import abc
from peng.utils.data_equal import DataEqualable
import collections
from collections import OrderedDict
from peng.utils import data_equal, data_reader
from peng.utils.common import _getArg

class CharData(DataEqualable,collections.Container,collections.Sized,collections.Iterable, abc.ABC):    
    ''' An abstract class to manipulate tabular data.'''  
    @abc.abstractproperty
    def _data(self):
        pass
    
    @abc.abstractmethod
    def convertKey(self,query):
        pass
    
    @abc.abstractproperty
    def name(self):
        pass
    
    def __contains__(self,key):
        return self.convertKey(key) in self._data
    
    def __len__(self):
        return len(self._data)
    
    def __iter__(self):
        return iter(self._data)
    
    def keys(self):
        return list(self._data.keys())
    
    def charNames(self):
        ans=[]
        ansSet=set()
        for k in self._data:
            for charName in self._data[k]:
                if not charName in ansSet:
                    ansSet.add(charName)
                    ans.append(charName)
        return ans
    
    def commonKeys(self,charNames=[],doFloat=False):
        if not charNames:
            charNames=self.charNames()
        ans=[]
        for k in list(self.keys()):
            okay=True
            for c in charNames:
                if self.get(k,c,doFloat)==None:
                    okay=False
                    break
            if okay:
                ans.append(k)
        return ans
    
    def commonChars(self,queries=[],doFloat=False):
        if not queries:
            queries=list(self.keys())
        ans=[]
        for c in self.charNames():
            okay=True
            for k in queries:
                if self.get(k,c,doFloat)==None:
                    okay=False
                    break
            if okay:
                ans.append(c)
        return ans
    
    def get(self, query, charName,doFloat=False):
        if not self.hasEntry(query,charName):
            return None
        k=self.convertKey(query)
        if doFloat:
            try:
                ans=float(self._data[k][charName])
            except:
                return None
            else:
                return ans
        else:
            return self._data[k][charName]
        
    def remove(self, query):
        k=self.convertKey(query)
        del self._data[k]
        
    def copy(self, target, query, charName, doFloat=False):
        if not isinstance(target,CharData):
            raise TypeError()
        if query in target:
            self.set(query,charName,target.get(query,charName,doFloat))
            
    def copyAll(self,target,query,charNames,doFloat=False):
        for charName in charNames:
            self.copy(target,query,charName,doFloat)
        
    def hasEntry(self,query,charName):
        k=self.convertKey(query)
        return k in self._data and charName in self._data[k]
    
    def set(self,query,charName,value):
        k=self.convertKey(query)
        if not k in self._data:
            self._data[k]=OrderedDict()
        self._data[k][charName]=value
        
    def dataEquals(self,obj):
        if not isinstance(obj,CharData):
            return False
        return data_equal.equal(self._data,obj._data)
    
    def getColumn(self,charName,doFloat=False,queries=[]):
        if not queries:
            queries=list(self.keys())
        ans=OrderedDict()
        for k in queries:
            ans[k]=self.get(k,charName,doFloat)
        return ans
    
    def setColumn(self,charName,data):
        for k in data:
            self.set(k,charName,data[k])
            
    def getRow(self,query,doFloat=False,charNames=[]):
        if not charNames:
            charNames=self.charNames()
        ans=OrderedDict()
        k=self.convertKey(query)
        for charName in charNames:
            ans[charName]=self.get(k,charName,doFloat)
        return ans
    
    def setRow(self,query,data):
        for charName in data:
            self.set(query,charName,data[charName])
            
    def passFilter(self,query,filter):
        if query in self:
            return data_reader.passFilter(self._data[query],*filter)
        return False
        
    def __str__(self):
        chars=self.charNames()
        lines=['%s with %d keys and %d columns'%(self.name,len(self),len(chars))]
        for k in self:
            lines.append('\t %s'%('\t'.join(['%s=%s'%(charName,self.get(k,charName)) for charName in chars])))
        return '\n'.join(lines)
    
class SimpleCharData(CharData):
    def __init__(self,**kwargs):
        self._name=_getArg(kwargs,'name','SimpleCharData')
        self.__data=OrderedDict()
    
    @property
    def name(self):
        return self._name
    
    @property
    def _data(self):
        return self.__data
    
    def convertKey(self,query):
        return query
        