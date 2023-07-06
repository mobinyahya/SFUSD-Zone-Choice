'''
Created on Nov 24, 2012

@author: pengshi
'''

import csv
from . import data_reader
from collections import OrderedDict

class CharFileReader(object):
    ''' A helper class to read in tabular data from CSV files.'''  
    def __init__(self,filename):
        self.filename=filename
        self.header=self.getHeader()
        
    def getHeader(self):
        inFile=open(self.filename)
        reader=csv.reader(inFile)
        header=next(reader)
        inFile.close()
        return header
      
    def __iter__(self):
        return self.iter()
        
    def iter(self,reqFields=[],reqValues=[],banFields=[],banValues=[]):
        assert len(reqFields)==len(reqValues) and len(banFields)==len(banValues),'length must match'
        inFile=open(self.filename)
        reader=csv.reader(inFile)
        next(reader)
        
        for row in reader:
            numInRow=len(row)
            ans=OrderedDict()
            for i in range(len(self.header)):
                h=self.header[i]
                # Read only the first entry
                if h not in ans:
                    if i<numInRow:
                        ans[h]=row[i].strip()
                    else:
                        ans[h]=""
            if data_reader.passFilter(ans, reqFields, reqValues, banFields, banValues):
                yield ans
        inFile.close()
        
    def getKeys(self,reqFields=[],reqValues=[],banFields=[],banValues=[]):
        ans=[]
        for line in self.iter(reqFields,reqValues,banFields,banValues):
            ans.append(line[self.header[0]])
        return ans
    
    def _convertToFloat(self,entry):
        try:
            ans=float(entry)
        except:
            ans=0
        return ans
    
    def getField(self,field,doFloat=True):
        ans=OrderedDict()
        id=self.header[0]
        for line in self.iter():
            if doFloat:
                ans[line[id]]=self._convertToFloat(line[field])
            else:
                ans[line[id]]=line[field]
        return ans
    