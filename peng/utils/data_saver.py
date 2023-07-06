'''
Created on Nov 24, 2012

Functions for writing data to files in various formats.

@author: pengshi
'''

import csv
import pickle
from peng.utils.common import _prepDir

class PickleSaveLoad(object):
    def save(self,filename):
        file=open(filename,'wb')
        pickle.dump(self, file, 2)
        file.close()
        
    @staticmethod
    def load(filename):
        file=open(filename,'rb')
        ans= pickle.load(file)
        file.close()
        return ans

def saveDict(filename,data,dictLevel,header=[]):
    _prepDir(filename)
    outFile=open(filename,'w')
    writer=csv.writer(outFile)
    if dictLevel<=1:
        if not header:
            header=['Key','Value']
        assert len(header)==2,'header size %d'%len(header)
        writer.writerow(header)
        for k in list(data.keys()):
            writer.writerow([k,data[k]])
    else:        
        if not header:
            header=['Key %d'%(l) for l in range(1,dictLevel)]
        lines,lastLevelKeys=_multiDictToList(data,dictLevel)
        writer.writerow(header+lastLevelKeys)
        for l in lines:
            writer.writerow(l)
    outFile.close()
    
def _multiDictToList(data,dictLevel):
    dicList=[]
    dicKeys=[]
    _recurseFlatten(data,dictLevel,1,dicList,dicKeys,[])
    lastLevelKeySet=set()
    lastLevelKeys=[]
    for l in dicList:
        for k in l:
            if not k in lastLevelKeySet:
                lastLevelKeySet.add(k)
                lastLevelKeys.append(k)
                
    ans=[]
    for i in range(len(dicList)):
        curValues=[dicList[i][k] if k in dicList[i] else "" for k in lastLevelKeys]
        ans.append(dicKeys[i]+curValues)
    return ans,lastLevelKeys

def _recurseFlatten(data,dictLevel,minLevel,dataList,keys,cur):
    if dictLevel<=minLevel:
        keys.append(list(cur))
        dataList.append(data)
        return
    for k in data:
        _recurseFlatten(data[k],dictLevel-1,minLevel,dataList,keys,cur+[k])
        
def saveLists(filename,data,dictLevel,header=[]):
    _prepDir(filename)
    outFile=open(filename,'w')
    writer=csv.writer(outFile)
    if not header:
        header=['Key %d'%l for l in range(1,dictLevel+1)]
    keys=[]
    lists=[]
    _recurseFlatten(data,dictLevel,0,lists,keys,[])
    writer.writerow(header+['Count','List'])
    for i in range(len(keys)):
        if len(lists[i]):
            writer.writerow(keys[i]+[len(lists[i])]+list(lists[i]))
    outFile.close()
    
def saveLines(filename,lines,header=[],sort=False):
    _prepDir(filename)
    outFile=open(filename,'w')
    writer=csv.writer(outFile)
    if header:
        writer.writerow(header)
    for row in lines:
        if sort:
            writer.writerow(sorted(row))
        else:
            writer.writerow(row)
    outFile.close()
    
def saveSet(filename,data,header=['Set']):
    _prepDir(filename)
    lines=[[a] for a in data]
    saveLines(filename, lines, header)
    
def saveChars(filename,data):
    _prepDir(filename)
    headers=[]
    headerSet=set()
    for _dict in data:
        for k in _dict:
            if k not in headerSet:
                headerSet.add(k)
                headers.append(k)
    lines=[]
    for _dict in data:
        line=[]
        for k in headers:
            line.append(_dict[k] if k in _dict else '')
        lines.append(line)
    saveLines(filename, lines, headers)
            