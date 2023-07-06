'''
Created on Jan 8, 2013

@author: pengshi
'''

import abc
from peng.utils.char_data import CharData
from peng.utils import data_saver
from peng.utils.char_reader import CharFileReader
from collections import OrderedDict
from peng.utils.common import _getArg

class CSVCharData(CharData, metaclass=abc.ABCMeta):
    ''' An abstract class to manipulate tabular data from CSV files. 
    It uses the methods in CharData but adds capabilities for reading from 
    and writing to CSV files.
    '''
    
    @abc.abstractproperty
    def keyHeaders(self):
        pass
    
    @abc.abstractmethod
    def keyToCSVRep(self,key):
        pass
    
    def toLines(self,skipKeys=False):
        charNames=self.charNames()
        if skipKeys:
            header=charNames
        else:
            header=self.keyHeaders+charNames
        lines=[]
        for k in self:
            if skipKeys:
                line=[]
            else:
                line=self.keyToCSVRep(k)
            for c in charNames:
                e=self.get(k,c)
                if e==None:
                    line.append('')
                else:
                    line.append(e)
            lines.append(line)
        return header,lines
    
    def saveCSV(self,filename,skipKeys=False):
        header,lines=self.toLines(skipKeys)
        data_saver.saveLines(filename,lines,header)
        return filename
        
    @staticmethod
    def setFromCSV(data,filename,keyHeaders,CSVRepToKeyFunc):
        for line in CharFileReader(filename):
            klist=[]
            for h in keyHeaders:
                klist.append(line[h])
                del line[h]
            k=CSVRepToKeyFunc(klist)
            for charName in line:
                data.set(k,charName,line[charName])
                
class SimpleCSVCharData(CSVCharData):
    ''' A basic implementation of CSVCharData.'''
    def __init__(self,**kwargs):
        self._name=_getArg(kwargs,'name','SimpleCSVCharData')
        self._key=_getArg(kwargs,'key','Key')
        self.__data=OrderedDict()
    
    @staticmethod
    def CSVRepToKey(rep):
        return rep[0]
    
    @staticmethod
    def loadCSV(filename,**kwargs):
        key=_getArg(kwargs,'key','Key')
        ans=SimpleCSVCharData(**kwargs)
        ans.setFromCSV(ans, filename, [key],ans.CSVRepToKey)
        return ans
    
    @property
    def name(self):
        return self._name
    
    @property
    def _data(self):
        return self.__data
    
    def convertKey(self,query):
        return query
        
    @property
    def keyHeaders(self):
        return [self._key]
    
    def keyToCSVRep(self,key):
        return [key]
    
    
    
