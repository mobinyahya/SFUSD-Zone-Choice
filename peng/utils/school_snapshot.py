'''
Created on Jan 8, 2013

@author: pengshi
'''
from peng.utils.csv_char_data import CSVCharData
from collections import OrderedDict
from peng.utils.common import _getArg
from peng.utils.program import ProgramCharHeader, ProgramId, Program

class SchoolSnapshot(CSVCharData):
    ''' A special data structure for storing school characteristics. '''
    def __init__(self,**kwargs):
        self._name=_getArg(kwargs,'name','SchoolSnapshot')
        self.__data=OrderedDict()
        
    @staticmethod
    def loadCSV(filename,**kwargs):
        data=SchoolSnapshot(**kwargs)
        data.setFromCSV(data, filename, data.keyHeaders, data.CSVRepToKey)
        return data
    
    @property
    def keyHeaders(self):
        return [ProgramCharHeader.SID]
    
    def CSVRepToKey(self,klist):
        return klist[0]
    
    def keyToCSVRep(self,key):
        return [key]
    
    @property
    def _data(self):
        return self.__data
    
    def convertKey(self,query):
        if isinstance(query,ProgramId):
            return query.sid
        elif isinstance(query,Program):
            return query.id.sid
        else:
            return str(query)
        
    @property
    def name(self):
        return self._name