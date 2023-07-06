'''
Created on Nov 24, 2012

Functions for reading in data from files of various formats.

@author: pengshi
'''
import csv
from collections import OrderedDict

def getWalkGeo(filename):
    reader=csv.reader(open(filename))
    next(reader)
    ans={}
    for row in reader:
        g,s=row
        g=g.strip()
        s=s.strip()
        if g not in ans:
            ans[g]=set()
        ans[g].add(s)
    return ans

def getGoogleDist(filename, inMiles=True):
    mile = 1609.34
    inFile = open(filename)
    reader = csv.reader(inFile)
    next(reader)
    ans = {}
    for row in reader:
        if (row[0] not in ans):
            ans[row[0]] = {}
        ans[row[0]][row[1]] = float(row[2])
        if (inMiles):
            ans[row[0]][row[1]] /= mile
    return ans

def _convertToFloat(entry):
    try:
        ans=float(entry)
    except:
        ans=0
    return ans

def getDict(filename, dictLevel, doFloat=True):
    inFile = open(filename)
    reader = csv.reader(inFile)
    ans = OrderedDict()
    if dictLevel == 1:
        next(reader)
        for row in reader:
            if doFloat:
                ans[row[0]] = _convertToFloat(row[1])
            else:
                ans[row[0]] = row[1]
    else:
        lastLevelKeys = next(reader)[dictLevel - 1:]
        for row in reader:
            prevKeys = row[:dictLevel - 1]
            lastLevelValues = row[dictLevel - 1:]
            if doFloat:
                lastLevelDict = OrderedDict({lastLevelKeys[j]:_convertToFloat(lastLevelValues[j]) for j in range(len(lastLevelKeys)) })
            else:
                lastLevelDict = OrderedDict({lastLevelKeys[j]:lastLevelValues[j].strip() for j in range(len(lastLevelKeys)) })
            _recursivePut(ans,lastLevelDict,prevKeys,0,dictLevel-2)
    inFile.close()
    return ans

def _recursivePut(bigDic,entry,keys,index,endIndex):
    if index>=endIndex:
        bigDic[keys[index]]=entry
        return
    #print bigDic, entry, keys, index,endIndex
    if keys[index] not in bigDic:
        bigDic[keys[index]]=OrderedDict()
    _recursivePut(bigDic[keys[index]],entry,keys,index+1,endIndex)
    
def getLists(filename,dictLevel,noHeader=False):
    inFile=open(filename)
    reader=csv.reader(inFile)
    if not noHeader:
        next(reader)
    ans=OrderedDict()
    for row in reader:
        keys=row[:dictLevel]
        count=int(row[dictLevel])
        curList=row[dictLevel+1:dictLevel+1+count]
        _recursivePut(ans,curList,keys,0,dictLevel-1)
    inFile.close()
    return ans

def getLines(filename,noIgnoreHeader=False):
    inFile=open(filename)
    reader=csv.reader(inFile)
    if not noIgnoreHeader:
        next(reader)
    ans=[]
    for row in reader:
        ans.append(row)
    inFile.close()
    return ans

def getSet(filename,noHeader=False):
    lines=getLines(filename,noHeader)
    ans={line[0] for line in lines}
    return ans

def passFilter(dict,reqFields,reqValues,banFields,banValues):
    debug=False
    for i,f in enumerate(reqFields):
        if f not in dict:
            if debug:
                print('no required key %s : %s'%(f,str(dict)))
            return False
        if not dict[f] in reqValues[i]:
            if debug:
                print('dict[%s]=%s not in required %s'%(f,dict[f],str(reqValues[i])))
            return False
    for i,f in enumerate(banFields):
        if f not in dict:
            if "" in banValues[i]:
                if debug:
                    print('no banned empty key %s: %s'%(f,str(dict)))
                return False
            continue
        if banValues[i] and dict[f] in banValues[i]:
            if debug:
                print('dict[%s]=%s is in banned %s'%(f,dict[f],str(banValues[i])))
            return False
    return True