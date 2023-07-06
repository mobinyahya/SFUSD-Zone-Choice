'''
Created on Oct 6, 2015

@author: pengshi
'''
import pandas as pd

from peng.assignment_plans.generic.plan import Plan
import abc
import numpy as np
from collections import Counter, OrderedDict
from peng.utils import data_saver, data_reader
from peng.utils.common import _getArg
from peng.assignment_plans.optimization.fast_logit import FastLogit
from peng.utils.csv_char_data import SimpleCSVCharData
import os
from peng.constants.locations import PlanFolders, PlanFiles
from peng.assignment_plans.generic.fixed_effect_utility import FixedEffectUtility

class RandomMenuPlan(Plan, metaclass=abc.ABCMeta):
    ''' An assignment plan based on giving each student a randomized budget set and
    having them choose their favorite within the set. This is the same as the 
    random assortment mechanism under the MNL model.
    
    The helper functions implemented here is used to evaluate the plan in the large market continuum model.    
    '''
    @property
    @abc.abstractmethod
    def utilParam(self):
        pass

    @abc.abstractmethod
    def cost(self,i,j):
        pass

    @property
    @abc.abstractmethod
    def randomMenuDict(self):
        pass
    
    def randomMenus(self,i):
        return self.randomMenuDict[i]
        
    @staticmethod
    def convertToCutoffs(randomMenuDict):
        ans={}
        for i in randomMenuDict:
            ans[i]=Counter()
            for M,prob in randomMenuDict[i]:
                for j in M:
                    ans[i][j]+=prob
        return ans
    
    @staticmethod
    def nestMenus(randomMenuDict):
        newDict={}
        for i in randomMenuDict:
            tmp=sorted([(len(M),-prob,j) for j,(M,prob) in enumerate(randomMenuDict[i])])
            curUnion=set()
            cur=[]
            for unused1,prob,j in tmp:
                curUnion|=randomMenuDict[i][j][0]
                cur.append((set(curUnion),-prob))
            newDict[i]=cur
        return newDict
    
    @staticmethod
    def optimisticMenus(utilParam,randomMenuDict):
        newDict={}
        for i in randomMenuDict:
            tmp=sorted([(utilParam.menuUtility(i,M),M) for M,prob in randomMenuDict[i]],reverse=True)
            newDict[i]=[(tmp[0][1],1)]
        return newDict
                
        
    
    def saveCutoffs(self,filename):
        ans=self.convertToCutoffs(self.randomMenuDict)
        data_saver.saveDict(filename, ans, 2)
        return filename
    
    def firstChoiceProb(self,i):
        M=self.choiceSet(i)
        choiceProb=self.utilParam.demandProb(i,M)

        ans=0
        for j in M:
            access=sum(prob for menu,prob in self.randomMenus(i) if j in menu)
            ans+=access*choiceProb[j]
        return ans
    
    def _computeCombinations(self,weights,denum):
        n=len(weights)
        tot=0
        for i in range(n):
            for j in range(n):
                if j==i:
                    continue
                for k in range(n):
                    if k==i or k==j:
                        continue
                    wi,wj,wk=weights[i],weights[j],weights[k]
                    tot+=wi/denum*wj/(denum-wi)*wk/(denum-wi-wj)
        return tot
    
    def top3ChoiceProb(self,i):
        choiceSet=self.choiceSet(i)
        denum=sum(self.utilParam.attractionWeight(i,j) for j in choiceSet)
        #print(f'Choice set for {i} is {choiceSet}')
        #print(f'\tDenum of {i} is {denum}')
        ans=0
        for M, prob in self.randomMenus(i):
            weights=[self.utilParam.attractionWeight(i,j) for j in choiceSet-set(M)]
            #print (f'\tMenu {M} with {prob}')
            #print(f'\t\tWeights:{weights}')
            ans+=prob*(1-self._computeCombinations(weights, denum))
        return ans            
    
    
    def averageFirstChoiceProb(self):
        return sum(self.firstChoiceProb(i)*self.weight(i) for i in self.iset)/float(self.totalWeights())
        
    def averageTop3ChoiceProb(self):
        return sum(self.top3ChoiceProb(i)*self.weight(i) for i in self.iset)/float(self.totalWeights())
    
    def objective(self):
        return self.totalUtility()-self.totalCost()
    
    def totalUtility(self):
        return sum(self.utility(i)*self.weight(i) for i in self.iset)
    
    def averageUtility(self):
        return self.totalUtility()/float(self.totalWeights())
    
    def averageUtilityCondAssigned(self):
        return sum(self.utility(i)/self.assignedProbability(i)*self.weight(i) for i in self.iset)/float(self.totalWeights())
    
    def minimumUtilityCondAssigned(self):
        return min(self.utility(i)/self.assignedProbability(i) for i in self.iset)

    def minimumUtilityTypeCondAssigned(self):
        return np.argmin([self.utility(i)/self.assignedProbability(i) for i in self.iset])
    
    def utility(self,i):
        return sum(prob*self.utilParam.menuUtility(i,M) for (M,prob) in self.randomMenus(i))+self.utilParam.utilityOffset(i)
    
    def minimumUtility(self):
        return min(self.utility(i) for i in self.iset)
    
    def minimumUtilityI(self):
        minU=np.Inf
        ans=None
        for i in self.iset:
            curU=self.utility(i)
            if curU<minU:
                minU=curU
                ans=i
        return ans
                
    def utilityQuantile(self,quantile):
        data=[self.utility(i) for i in self.iset]
        return np.percentile(data,quantile)
    

    
    def totalCost(self):
        ans=0
        for i in self.iset:
            for j in self.choiceSet(i):
                ans+=self.weight(i)*self.assignmentProbability(i,j)*self.cost(i,j)
        return ans
    
    def averageCost(self):
        return self.totalCost()/float(self.totalWeights())
    
    def assignmentProbability(self,i,j):    
        ans=0
        for M,prob in self.randomMenus(i):
            if j not in M:
                break
            ans+=prob*self.utilParam.demandProb(i,M)[j]
        return ans
    
    def assignedProbability(self,i):
        return sum(prob for menu,prob in self.randomMenus(i))
    
    def totalAssignmentProb(self):
        ans=0
        for i in self.iset:
            ans+=self.weight(i)*self.assignedProbability(i)
        return ans
    
    def averageAssignmentProb(self):
        return self.totalAssignmentProb()/float(self.totalWeights())
    
    
    def dataEquals(self,obj):
        if not isinstance(obj,RandomMenuPlan):
            return False
        if not super(RandomMenuPlan,self).dataEquals(obj):
            return False
        for i in self.iset:
            myMenus=sorted(self.randomMenus(i))
            otherMenus=sorted(obj.randomMenus(i))
            if not myMenus==otherMenus:
                return False
        return True    
    
    def saveRandomMenus(self,outFile,writeChoiceProbs=False):
        jList=sorted(self.totalJSet())
        jDict={j:ind for ind,j in enumerate(jList)}
        header=['Type','Prob','Value']+jList
        lines=[]
        S=len(jList)
        for i in self.iset:
            for M,prob in self.randomMenus(i):
                vec=np.zeros(S)
                MInds=[jDict[j] for j in M]
                if not writeChoiceProbs:
                    vec[MInds]=1
                else:
                    probs=self.utilParam.demandProb(i,M)
                    for j in probs:
                        vec[jDict[j]]=probs[j]
                line=[i,prob,self.utilParam.menuUtility(i,M)+self.utilParam.utilityOffset(i)]+list(vec)
                lines.append(line)
            
        data_saver.saveLines(outFile,lines,header)
        

    # def saveExpectedCapacities(self,outFile):
    #     ans=self.expectedCapacities()
    #     data_saver.saveDict(outFile,ans,1)
        
    @staticmethod
    def readMenus(inFile):
        lines=data_reader.getLines(inFile, True)
        header=lines[0]
        jSet=header[3:]
        ans=OrderedDict()
        for line in lines[1:]:
            i=line[0]
            prob=float(line[1])
            menu=set()
            for j,value in enumerate(line[3:]):
                if float(value)>0:
                    menu.add(jSet[j])
            if i not in ans:
                ans[i]=[]
            ans[i].append((menu,prob))
            
        return ans
            
    def outputAnalysis(self,outFile):
        out=SimpleCSVCharData(key='Type')
        for i in self.iset:
            out.set(i,'Weight',self.weight(i))
            out.set(i,'Menu Size',len(self.choiceSet(i)))
            out.set(i,'Busing Menu Size',len([s for s in self.choiceSet(i) if self.cost(i,s)>0]))
            out.set(i,'Utility',self.utility(i))
            out.set(i,'firstChoiceProb',self.firstChoiceProb(i))
            out.set(i,'Assignment Prob',self.assignedProbability(i))
        out.saveCSV(outFile)

    # def averageBusingArea(self):
    #     geoChar=SimpleCSVCharData.loadCSV(os.path.join(PlanFiles.GEO_CHAR),key='Geocode')
    #     num=0
    #     for geo in self.iset:
    #         if geo in geoChar:
    #             num+=geoChar.get(geo, 'Area', True)*len([j for j in self.choiceSet(geo) if self.cost(geo,j)>0])
    #     num/=float(len(self.totalJSet()))
    #     return num
    #
    # def averageBusingChoices(self):
    #     geoChar=SimpleCSVCharData.loadCSV(os.path.join(PlanFiles.GEO_CHAR),key='Geocode')
    #     num=0
    #     denum=0
    #     for geo in self.iset:
    #         if geo in geoChar:
    #             num+=geoChar.get(geo,'Weight',True)*len([j for j in self.choiceSet(geo) if self.cost(geo,j)>0])
    #             denum+=geoChar.get(geo,'Weight',True)
    #     return num/denum
    
    
        
    @staticmethod
    def inferMenusFromCutoffs(cutoffs):
        randomMenuDict={}
        for i in cutoffs:
            tmp=sorted([(cutoffs[i][j],j) for j in cutoffs[i]])
            cur=[]
            if len(tmp)==0:
                return cur
            if tmp[0][0]>0:
                cur.append((set([tmp[j][1] for j in range(len(tmp))]),tmp[0][0]))
            for k in range(1,len(tmp)):
                prob=tmp[k][0]-tmp[k-1][0]
                if prob>0:
                    cur.append((set([tmp[j][1] for j in range(k,len(tmp))]),prob))
            randomMenuDict[i]=cur
        return randomMenuDict
        
            
class SimpleRandomMenuPlan(RandomMenuPlan):
    def __init__(self,randomMenuDict,weightDict,utilParam,costDict,**kwargs):
        self._iset=set(randomMenuDict.keys())
        self._name=_getArg(kwargs,'name','SimpleRandomMenuPlan')
        
        self._choiceSet={}
        self._weightDict={}
        
        self._randomMenuDict={}
        for i in randomMenuDict:
            self._randomMenuDict[i]=[(set(M),prob) for M,prob in randomMenuDict[i]]

        self._weightDict = weightDict
        for i in self._iset:
            # if i in weightDict:
            #     self._weightDict[i]=weightDict[i]
            # else:
            #     self._weightDict[i]=0
            self._choiceSet[i]=set()
            for M,prob in self._randomMenuDict[i]:
                self._choiceSet[i]|=M
                
        self._utilParam=utilParam
        self._cost={}
        self._capSchools=[159]

        # for i in self._iset:
        #     self._cost[i]=Counter()
        #     # if i not in self._capSchools:
        #     #     self._capSchools[i]={}
        #     for j in self._choiceSet[i]:
        #         if i in costDict and j in costDict[i]:
        #             self._cost[i][j]=costDict[i][j]
        #         else:
        #             self._cost[i][j]=0
        self._cost = costDict

        self._utility,self._assignmentProb=self._cacheVP()
        
    def expectedCapacity(self,j):
        return sum(self.weight(i)*self.assignmentProbability(i,j) for i in self.iset if j!=self._capSchools)
    
    def expectedCapacities(self):
        return Counter({j: self.expectedCapacity(j) for j in self.totalJSet()})
    
    @staticmethod    
    def planFromName(name,convertCutoffs=False,useNewYear=False):
        # if useNewYear:
        #     utilParam=FixedEffectUtility.simple2014K12()
        # else:
        #     utilParam=FixedEffectUtility.simple2013K12()
        utilParam = FixedEffectUtility.sfusd2018K()
            
        # weightDict=data_reader.getDict(PlanFiles.GEO_CHAR,1,True)
        # costDict=data_reader.getDict(PlanFiles.MILES_BUSED,2,True)
        weights = pd.read_csv(PlanFiles.STUDENT_COUNTS, index_col=0)
        weightDict = dict(zip(weights.index, weights.student_count))
        costDict = pd.read_csv(PlanFiles.DISTANCES, index_col=[0, 1]).values
        costDict = np.hstack([costDict, np.zeros((costDict.shape[0], 1))])
        
        randomMenuFile=os.path.join(PlanFolders.PRIORITIES,name,'menus.csv')
        if os.path.exists(randomMenuFile):
            randomMenuDict=RandomMenuPlan.readMenus(randomMenuFile)
            if convertCutoffs:
                cutoffs=RandomMenuPlan.convertToCutoffs(randomMenuDict)
                randomMenuDict=RandomMenuPlan.inferMenusFromCutoffs(cutoffs)
        else:
            cutoffs=data_reader.getDict(os.path.join(PlanFolders.PRIORITIES,'%s.csv'%name), 2, True)
            randomMenuDict=RandomMenuPlan.inferMenusFromCutoffs(cutoffs)
        return SimpleRandomMenuPlan(randomMenuDict,weightDict,utilParam,costDict)
    
    @property
    def randomMenuDict(self):
        return self._randomMenuDict
    
    def _cacheVP(self):
        logit=FastLogit.fromUtilParam(self.utilParam)
        v=np.zeros(logit.T)
        p=np.zeros((logit.T,logit.S))
        beta=logit.beta
        euler=0.577215664901532

        for i in self.iset:
            t=logit.iind(int(i))
            for (menu,prob) in self.randomMenus(i):
                M=[logit.jind(int(j)) for j in menu]
                curv,curp=logit.vp(t, M)
                v[t]+=curv*prob
                p[t,M]+=curp*prob
            v[t]+=euler*beta[t]
        utility={}
        assignmentProb={}
        for i in self.iset:
            t=logit.iind(int(i))
            utility[i]=v[t]
            assignmentProb[i]={}
            for j in self.choiceSet(i):
                assignmentProb[i][j]=p[t,logit.jind(int(j))]
        return utility,assignmentProb
    
    def utility(self,i):
        return self._utility[i]
    
    def assignmentProbability(self,i,j):
        if not j in self._assignmentProb[i]:
            return 0
        else:
            return self._assignmentProb[i][j]
    
    @property
    def iset(self):
        return self._iset
    
    def choiceSet(self,i):
        return self._choiceSet[i]
    
    def weight(self,i):
        return self._weightDict[int(i)]
    
    @property
    def name(self):
        return self._name
    
    @property
    def utilParam(self):
        return self._utilParam
    
    def cost(self,i,j):
        return self._cost[int(i)][int(j)]
    
    
    @staticmethod
    def fromFile(randomMenuFile,weightFile,utilParam,costFile,**kwargs):
        randomMenuDict=RandomMenuPlan.readMenus(randomMenuFile)
        weightDict=data_reader.getDict(weightFile,1,True)
        costDict=data_reader.getDict(costFile,2,True)
        return SimpleRandomMenuPlan(randomMenuDict,weightDict,utilParam,costDict,**kwargs)
    
