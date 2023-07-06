'''
Created on Feb 5, 2014

@author: pengshi
'''
from peng.constants.locations import PlanFolders, ZoneFiles, PlanFiles, Folders
import os
from peng.utils.csv_char_data import SimpleCSVCharData
from peng.assignment_plans.generic.fixed_effect_utility import FixedEffectUtility

from peng.deferred_acceptance.type_dastb import SimpleBin, FastLogitChooser, TypeDASTB
from collections import Counter, OrderedDict
from peng.utils.weighted_random import WeightedRandom
from peng.utils.common import _getArg
import time

import random

import numpy as np
from peng.utils import data_reader
from peng.utils.geo_util import GeoUtil
from peng.utils.distance_cache import DistanceCache
from peng.utils.school_snapshot import SchoolSnapshot
from peng.deferred_acceptance.assignment import Assignment
from peng.deferred_acceptance.old_boston_dastb import OldBostonDASTB

class FiniteMarketSimulator(object):
    ''' Simulation engine in the finite market stochastic model. 
    This is used to evaluate all the assignment plans and make the result tables.'''
    def __init__(self,use2014Params=False,mult=1,totalVariation=True, regionalVariation=True, useCapFile='',capSchoolFile=''):
        self._simDir=PlanFolders.SIMULATIONS
        
        self._mult=mult
        self._hasVariance=totalVariation
        self._regionalVariation=regionalVariation
        self._hoodGeos,self._geoWeights=self._generateGeoWeights()
        
        self._newRatio=SimpleCSVCharData.loadCSV(PlanFiles.REGION_RATIO,key='Neighborhood')
        self._use2014Params=use2014Params
        
        if use2014Params:
            self._utilityParam=FixedEffectUtility.simple2014K12()
            
        else:
            if totalVariation:
                self._totMean=4294*mult
                self._totStd=115*mult
            else:
                self._totMean=4297.9096422818*mult
            self._utilityParam=FixedEffectUtility.simple2013K12()
            
        self._schoolChar=SimpleCSVCharData.loadCSV(PlanFiles.FIXED_EFFECT,key='School Code')
        if capSchoolFile:
            self._capSchools=data_reader.getDict(capSchoolFile,1,False)
        else:
            self._capSchools=data_reader.getDict(PlanFiles.CAP_SCHOOLS,1,False)
        
        self._useCapFile=useCapFile
        self._capCode='CAP'
        self._dc=DistanceCache()
        self._geos=[str(intGeo) for intGeo in range(868)]
        
        self._geoChar=SimpleCSVCharData.loadCSV(PlanFiles.GEO_CHAR,key='Geocode')
        
    def _generateGeoWeights(self):
        geoHood=data_reader.getDict(ZoneFiles.NEIGHBORHOOD_NEW,1,False)
        tallies=data_reader.getDict(PlanFiles.SAMPLING_WEIGHTS,1,True)
        hoodGeos={}
        hoodSums=Counter()
        for geo in geoHood:
            hood=geoHood[geo]
            if not hood in hoodGeos:
                hoodGeos[hood]=[]
            if geo in tallies:
                hoodGeos[hood].append(geo)
                hoodSums[hood]+=tallies[geo]
        geoWeights={}
        for hood in hoodGeos:
            geoWeights[hood]=[]
            for geo in hoodGeos[hood]:
                geoWeights[hood].append(tallies[geo]/hoodSums[hood])
        return hoodGeos,geoWeights
        
    def _generateNeighbors(self):
        ans={}
        geoUtil=GeoUtil()
        for geo in self._geos:
            ans[geo]=[]
            for nei in self._geos:
                if geo==nei:
                    continue
                if geoUtil.walkDist(geo, nei)<=0.5:
                    ans[geo].append(nei)
        return ans
                
    
    def _allOptions(self,rawMenus):
        sids=set()
        for geo in rawMenus:
            for sid in rawMenus[geo]:
                sids.add(sid)
        return sorted(sids)
        
    def _makeTypePrio(self,params):
        ans={}
        for geo in params:
            ans[geo]={}
            capSid=self._capSchools[geo]
            for sid in params[geo]:
                prio=params[geo][sid]['Priority']
                if sid==capSid:
                    ans[geo][self._capCode]=prio
                else:
                    ans[geo][sid]=prio
        return ans
        
        
    def _processLogitBase(self,rawMenus,sidBin):
        
        baseUtilities={}
        menus={}
        for geo in rawMenus:
            tempMenu=[]
            tmpUtilities=[]
            capSid=self._capSchools[geo]
            for sid in rawMenus[geo]:
                realSid=sid
                if sid==capSid:
                    sid=self._capCode
                tempMenu.append(sidBin[sid])
                tmpUtilities.append(self._utilityParam.baseUtility(geo, realSid))
            baseUtilities[geo]=np.array(tmpUtilities)
            menus[geo]=np.array(tempMenu)
        return baseUtilities,menus
            
    def _makeBins(self,sids):
        ans=[]
        charFile=PlanFiles.FIXED_EFFECT
        if self._useCapFile:
            print('Using capacity file %s'%self._useCapFile)
            caps=data_reader.getDict(self._useCapFile,1,True)
        else:
            caps=SchoolSnapshot.loadCSV(charFile).getColumn('K2 Capacity', True)
        
        for sid in sids:
            capacity=caps[sid]*self._mult
            ans.append(SimpleBin(sid,capacity))
        ans.append(SimpleBin(self._capCode,1e10))
        return ans
            
    def _generateGeoCount(self):
        if self._hasVariance:
            tot=max(0,np.random.normal(self._totMean,self._totStd))
        else:
            tot=self._totMean
            
        print('Total: ', tot)
        print('RegionalVariation: ', self._regionalVariation)
        
        cnt=Counter()
        if self._regionalVariation:
            for hood in self._newRatio:
                hoodMean=self._newRatio.get(hood,'Mean',True)
                hoodStd=self._newRatio.get(hood,'Standard Deviation',True)
                ratio=max(0,np.random.normal(hoodMean,hoodStd))
                hoodNum=int(round(ratio*tot))
                for i in range(hoodNum):
                    geo=WeightedRandom.getFromWeights(self._hoodGeos[hood], self._geoWeights[hood])
                    cnt[geo]+=1
        else:
            weights=self._geoChar.getColumn('Weight', True)
            for geo in weights:
                raw=weights[geo]*self._mult
                fl=int(raw)
                if random.random()<raw-fl:
                    cnt[geo]=fl+1
                else:
                    cnt[geo]=fl
            
        ans=OrderedDict()
        for geo in self._geos:
            if cnt[geo]>0:
                ans[geo]=cnt[geo]
        return ans
    
    def _get2014GeoCount(self):
        filename=os.path.join(Folders.INPUTS,'geocode_sampling_weights_2014.csv')
        ans=data_reader.getDict(filename, 1, True)
        for t in ans:
            ans[t]=int(ans[t])
        return ans
    
                
    def _cleanAssignment(self,assignment):
        ans=Assignment()
        for chooser in assignment:
            sid=assignment[chooser].identity
            ans[chooser.identity]=sid
        return ans
        
    def _getSid(self,option,geo):
        if option==self._capCode:
            return self._capSchools[geo]
        else:
            return option
        
    def _getRawMenusFromParam(self,params):
        menus={}
        for geo in params:
            menus[geo]=[sid for sid in params[geo]]
        return menus
        
    def simulate(self,name,paramFile,N,oldWalkPrio=False,**kwargs):
        baseDir=os.path.join(self._simDir,name)
        outFile=os.path.join(baseDir,'Averages.csv')
        
        startTime=time.time()
        
        ansDenum=Counter()
        ansUtility=Counter()
        ansTop1=Counter()
        ansTop3=Counter()
        ansDist=Counter()
        ansBusing=Counter()
        ansCohesion=Counter()
        
        params=data_reader.getDict(paramFile, 3,True)
        rawMenus=self._getRawMenusFromParam(params)
        typePrio=self._makeTypePrio(params)
        
            
        sids=self._allOptions(rawMenus)
            
        sidInd=dict((sid,i) for i,sid in enumerate(sids))
        
        bins=self._makeBins(sids)
        sidBin={bin.identity:bin for bin in bins}
        accessNum={geo:Counter() for geo in self._geos}
        accessDenum={geo:Counter() for geo in self._geos}
        
        neighbors=self._generateNeighbors()
        
        baseUtilities,menus=self._processLogitBase(rawMenus,sidBin)
        
        cost=data_reader.getDict(PlanFiles.MILES_BUSED,2,True)
        
        print('Done preping in time %.2f'%(time.time()-startTime))
        
        for ind in range(N):
            innerTime=time.time()
            print('Simulating round %d'%ind) 
            if self._use2014Params:
                numGeo=self._get2014GeoCount()
            else:
                numGeo=self._generateGeoCount()
            choosers=[]
            geoInds={}
            i=0
            for geo in numGeo:
                geoInds[geo]=[]
                for j in range(numGeo[geo]):
                    choosers.append(FastLogitChooser(i,geo,baseUtilities[geo],menus[geo],self._utilityParam.idioSize(geo)))
                    geoInds[geo].append(i)
                    i+=1
                    
            print('\t done generating choices in %.2f'%(time.time()-innerTime))
            
#            for chooser in choosers:
#                print chooser
            if oldWalkPrio:
                da=OldBostonDASTB(choosers,bins,typePrio)
            else:
                da=TypeDASTB(choosers,bins,typePrio)

            print('\t done assignment in %.2f'%(time.time()-innerTime))
            print('\t\t Num unassigned %.0f'%(da.numUnassigned()))
            assignment=self._cleanAssignment(da.assign())
            
#            for bin in bins:
#                print bin.identity, da.getCutoff(bin)
            

            
            utility=Counter()
            dist=Counter()
            top1=Counter()
            top3=Counter()
            busing=Counter()
            assignVec={}
            for geo in numGeo:
                assignVec[geo]=np.zeros(len(sids))
                for i in geoInds[geo]:
                    optionSid=assignment[i]
                    option=sidBin[optionSid]
                    myUtil=choosers[i].utility(option)
                    
                    utility[geo]+=myUtil
                    
                    choiceNum=choosers[i].choiceNum(option)
                    top1[geo]+=(choiceNum<=1)
                    top3[geo]+=(choiceNum<=3)
                    sid=self._getSid(optionSid, geo)
                    curDist=self._dc.walkDist(2014, geo, sid)
                    dist[geo]+=curDist
                    busing[geo]+=cost[geo][sid]
                    assignVec[geo][sidInd[sid]]+=1
                    
            
                for optionSid in typePrio[geo]:
                    option=sidBin[optionSid]
                    if option.capacity>0:
                        sid=self._getSid(optionSid, geo)
                        accessNum[geo][sid]+=da.getAccess(option,geo) #curAccess
                        accessDenum[geo][sid]+=1
                    
                ansDenum[geo]+=float(numGeo[geo])
                ansUtility[geo]+=utility[geo]
                ansTop1[geo]+=top1[geo]
                ansTop3[geo]+=top3[geo]
                ansDist[geo]+=dist[geo]
                ansBusing[geo]+=busing[geo]
            
            for geo in numGeo:
                ansCohesion[geo]+=((np.dot(assignVec[geo],assignVec[geo])+sum(np.dot(assignVec[geo],assignVec[nei]) for nei in neighbors[geo] if nei in assignVec))-numGeo[geo])
                
            #print '\t done calculating in %.2f'%(time.time()-innerTime)
            print('\t cur time %.2f'%(time.time()-startTime))
            
        ans=SimpleCSVCharData(key='Geocode')
        
        numChoices={geo:len(rawMenus[geo]) for geo in ansDenum}
        numBusingChoices={geo:sum(cost[geo][s]>0 for s in rawMenus[geo]) for geo in rawMenus} 
        weight={geo:ansDenum[geo]/float(N) for geo in ansDenum}
        area={geo:self._geoChar.get(geo,'Area',True) for geo in rawMenus}
        for geo in area:
            if area[geo] is None:
                area[geo]=0
        
        for geo in self._geos:
            if ansDenum[geo]>0:
                ans.set(geo,'Distance',ansDist[geo]/ansDenum[geo])
                ans.set(geo,'Busing',ansBusing[geo]/ansDenum[geo])
                ans.set(geo,'Utility',ansUtility[geo]/ansDenum[geo])
                ans.set(geo,'Top1',ansTop1[geo]/ansDenum[geo])
                ans.set(geo,'Top3',ansTop3[geo]/ansDenum[geo])
                ans.set(geo,'Cohesion',ansCohesion[geo]/ansDenum[geo])
                ans.set(geo,'NumPeople',ansDenum[geo]/float(N))
                ans.set(geo,'NumChoices',numChoices[geo])
                ans.set(geo,'NumBusingChoices',numBusingChoices[geo])
                ans.set(geo,'Weight',weight[geo])
                ans.set(geo,'Area',area[geo])
                
        totDenum=float(sum(ansDenum[geo] for geo in ansDenum))
        ans.set('Average','Distance',sum(ansDist[geo] for geo in ansDenum)/totDenum)
        ans.set('Average','Busing',sum(ansBusing[geo] for geo in ansDenum)/totDenum)
        ans.set('Average','Utility',sum(ansUtility[geo] for geo in ansDenum)/totDenum)
        utilList=[ansUtility[geo]/ansDenum[geo] for geo in ansDenum]
        ans.set('5 pct','Utility',np.percentile(utilList,5))
        ans.set('10 pct','Utility',np.percentile(utilList,10))
        ans.set('Median','Utility',np.percentile(utilList,50))
        
        ans.set('Average','Top1',sum(ansTop1[geo] for geo in ansDenum)/totDenum)
        ans.set('Average','Top3',sum(ansTop3[geo] for geo in ansDenum)/totDenum)
        ans.set('Average','Cohesion',sum(ansCohesion[geo] for geo in ansDenum)/totDenum)
        
        cohesionList=[ansCohesion[geo]/ansDenum[geo] for geo in ansDenum]
        ans.set('5 pct','Cohesion',np.percentile(cohesionList,5))
        ans.set('10 pct','Cohesion',np.percentile(cohesionList,10))
        ans.set('Median','Cohesion',np.percentile(cohesionList,50))
        
        ans.set('Average','NumPeople',sum(ansDenum[geo] for geo in ansDenum)/float(len(ansDenum)*N))
        ans.set('Average','NumChoices',sum(numChoices[geo]*weight[geo] for geo in weight)/sum(weight[geo] for geo in weight))
        ans.set('Average','NumBusingChoices',sum(numBusingChoices[geo]*weight[geo] for geo in weight)/sum(weight[geo] for geo in weight))
        ans.set('Average','Weight',sum(weight[geo] for geo in weight))        
        ans.set('Average','Area',sum(area[geo] for geo in area))
        ans.set('Average','BusingArea',sum(numBusingChoices[geo]*area[geo] for geo in rawMenus)/float(len(sids)))
        
        ans.saveCSV(outFile)
               
        self.outAccess(os.path.join(baseDir,'Access.csv'),accessNum,accessDenum)
            
            
        print('\t done all time %.2f'%(time.time()-startTime))
        
    def outAccess(self,outFile,accessNum,accessDenum):
        out=SimpleCSVCharData(key='Geocode')
        for geo in self._geos:
            for sid in accessNum[geo]:
                cur=accessNum[geo][sid]/accessDenum[geo][sid]
                out.set(geo,sid,cur)
        out.saveCSV(outFile)
        

        
    def analyze(self,outFile,name,**kwargs):
        geoFile=SimpleCSVCharData.loadCSV(os.path.join(self._simDir,name,'Averages.csv'),key='Geocode')
        if _getArg(kwargs,'new',True) or not os.path.exists(outFile):
            ans=SimpleCSVCharData(key='Metric')
        else:
            ans=SimpleCSVCharData.loadCSV(outFile,key='Metric')
        
        ans.set('(1) Av. # of choices',name,geoFile.get('Average','NumChoices'))            
        ans.set('(2) Av. miles to assigned school',name,geoFile.get('Average','Distance'))
        ans.set('(3) Miles bused per student',name,geoFile.get('Average','Busing'))
        ans.set('(4) Av. bus coverage area',name,geoFile.get('Average','BusingArea'))
        ans.set('(5) Av. # of busing choices',name,geoFile.get('Average','NumBusingChoices'))
        
        ans.set('(6) Weighted average utility',name,geoFile.get('Average','Utility'))
        #ans.set('5 pct utility',name,geoFile.get('5 pct','Utility'))
        ans.set('(7) 10th percentile utility',name,geoFile.get('10 pct','Utility'))
        minUtil=np.Inf
        worstGeo=0
        for geo in self._geos:
            util=geoFile.get(geo,'Utility')
            if util:
                if float(util)<minUtil:
                    minUtil=float(util)
                    worstGeo=geo
        ans.set('(8) Lowest utility of any neighborhood',name,minUtil)
        #ans.set('Worst off type',name,worstGeo)
        ans.set('(9) % getting top 1 choice in menu',name,geoFile.get('Average','Top1'))
        ans.set('(10) % getting top 3 choice in menu',name,geoFile.get('Average','Top3'))
        
        #ans.set('Av. # of neighbors co-assigned',name,geoFile.get('Average','Cohesion'))
        #ans.set('Median # of neighbors co-assigned',name,geoFile.get('Median','Cohesion'))
        
        #ans.set('Max \# Choices',name,max(geoFile.get(geo,'NumChoices',True) for geo in self._geos if geo in geoFile))
        
        
        
        
        ans.saveCSV(outFile)
    
