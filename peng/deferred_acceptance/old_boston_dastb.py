'''
Created on Jul 15, 2019

@author: pengshi
'''

from peng.deferred_acceptance.type_dastb import SimpleBin,TypeDASTB
from peng.deferred_acceptance.chooser import SimpleTypeChooser
from peng.deferred_acceptance.assignment import Assignment
import math

class OldBostonDASTB(object):
    ''' Implements the DA algorithm under the priorities of the 3-Zone plan, in which
    each school is divided into a walk-half and an open-half. The preferences of students
    are expanded to be over halves, so that students in the walk-zone apply to the open-half
    first and those outside apply to the open-half first. The walk-half prioritizes students in the
    walk-zone over those outside, and the open-half does not respect walk-zone priorities.'''
    def __init__(self,choosers,bins,typePrio):
        newBins,self._walkMap,self._openMap,self._reverseMap=self._makeBins(bins)
        newChoosers=self._makeChoosers(choosers,self._walkMap,self._openMap,typePrio)
        self._newTypePrio=self._makeTypePrio(typePrio)
        self._da=TypeDASTB(newChoosers,newBins,self._newTypePrio)
#        print(newBins)
        
    def assign(self):
        ass=self._da.assign()
        newAssign=Assignment()
        for c in ass:
            b=ass[c]
            if b is None:
                newAssign[c]=None
            else:
                newAssign[c]=self._reverseMap[b]
                
        return newAssign
    
    def numUnassigned(self):
        return self._da.numUnassigned()
    
    def getAccess(self,b,Type):
        bw=self._walkMap[b]
        bo=self._openMap[b]
        wCutoff=self._da.getCutoff(bw)
        oCutoff=self._da.getCutoff(bo)
        if wCutoff is None:
            wAccess=0
        else:
            wAccess=min(max(1+self._newTypePrio[Type][bw.identity]-wCutoff,0),1)

        if oCutoff is None:
            oAccess=0
        else:
            oAccess=min(max(1+self._newTypePrio[Type][bo.identity]-oCutoff,0),1)
        access=max(wAccess,oAccess)
        return access


    def _makeBins(self,bins):

        walkMap={}
        openMap={}
        reverseMap={}
        newBins=[]
        for b in bins:
            name=str(b.identity)
            capacity=b.capacity
            bCap=int(math.ceil(capacity/2))
            bw=SimpleBin(name+'-Walk',bCap)
            bo=SimpleBin(name+'-Open',capacity-bCap)
            newBins.append(bw)
            newBins.append(bo)
            walkMap[b]=bw
            openMap[b]=bo
            reverseMap[bw]=b
            reverseMap[bo]=b
        return newBins,walkMap,openMap,reverseMap
    
    def _makeChoosers(self,choosers,walkMap,openMap,typePrio):
        newChoosers=[]
        for c in choosers:
            choices=[]
            for b in c.choices:
                if typePrio[c.type][b.identity]:
                    choices.append(walkMap[b])
                    choices.append(openMap[b])
                else:
                    choices.append(openMap[b])
                    choices.append(walkMap[b])
            newChoosers.append(SimpleTypeChooser(c.identity,c.type,choices))
        return newChoosers
    
    def _makeTypePrio(self,typePrio):
        newTypePrio={}
        for t in typePrio:
            newTypePrio[t]={}
            for sid in typePrio[t]:
                newTypePrio[t][sid+'-Walk']=typePrio[t][sid]
                newTypePrio[t][sid+'-Open']=0
        return newTypePrio