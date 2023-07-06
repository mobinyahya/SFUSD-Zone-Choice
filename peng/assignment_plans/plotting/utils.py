'''
Created on Oct 26, 2015

@author: pengshi
'''
import os
from peng.constants.locations import PlanFolders, PlanFiles
from peng.utils import data_reader
from collections import Counter
from peng.utils.geo_util import GeoUtil
import matplotlib.pyplot as plt
from peng.utils.csv_char_data import SimpleCSVCharData
from peng.utils.school_snapshot import SchoolSnapshot
from peng.utils.plan_visualizer import PlanVisualizer
from peng.utils.program import ProgramCharHeader
from peng.utils.common import _prepDir

def plotQuotas(planName,mult1=1,mult2=1,outFile1='',outFile2=''):
    ''' Plot the expected endowment and allowance of number of schools across Boston'''
    quotaFile=os.path.join(PlanFolders.PRIORITIES,planName,'quota.csv')
    shadowPrice=data_reader.getDict(os.path.join(PlanFolders.PRIORITIES,planName,'shadowPrice.csv'),1,True)
    busing=shadowPrice['BusingDistance']
    
    lines=data_reader.getLines(quotaFile)
    avQuota=Counter()
    avCardinality=Counter()
    for geo, prob, quota, card in lines:
        avQuota[geo]+=float(prob)*float(quota)
        avCardinality[geo]+=float(prob)*float(card)
    for geo in avQuota:
        avQuota[geo]=min(100,avQuota[geo])/busing
        #avQuota[geo]=avQuota[geo]/busing
    GeoUtil().plotGeoDict(avQuota, 0, mult1, outFile1)
    GeoUtil().plotGeoDict(avCardinality, 0, mult2, outFile2)
    
def plotSchoolCost(planName,mult=1,outFile=''):
    ''' Plot the cost of each school based on the shadow price of the capacity constraints'''
    shadowPrice=data_reader.getDict(os.path.join(PlanFolders.PRIORITIES,planName,'shadowPrice.csv'),1,True)
    busing=shadowPrice['BusingDistance']
    shadowPrice.pop('BusingDistance')
    shadowPrice.pop('BusingArea')
    shadowPrice.pop('BusingCardinality')
    ans=Counter()
    for sid in shadowPrice:
        ans[sid]=shadowPrice[sid]/busing
    plotSchoolStats(ans,mult,outFile)
    
    
def plotField(simName,fieldName,minAns=0,mult=1,outFile=''):
    ''' Plot a characteristic of the geocodes (neighborhoods).'''
    inFile=os.path.join(PlanFolders.SIMULATIONS,simName,'Averages.csv')
    inData=SimpleCSVCharData.loadCSV(inFile,key='Geocode')
    ans=Counter()
    for geo in range(868):
        stGeo=str(geo)
        if stGeo in inData:
            ans[stGeo]=inData.get(stGeo,fieldName,True)
    GeoUtil().plotGeoDict(ans, minAns,mult, outFile)


def plotSchoolStats(schoolValues,mult,outFile=''):
    ''' Plot a characteristic of the schools.'''
    filename=os.path.join(PlanFiles.FIXED_EFFECT)
    schoolChar=SchoolSnapshot.loadCSV(filename)
    v=PlanVisualizer()
    fig=plt.figure(figsize=(12,16))
    ax=plt.subplot(111)
    v.prepareCanvas(ax)
    H=ProgramCharHeader
    
    sids=schoolValues.keys()
    labels={schoolChar.get(sid,H.GEO):schoolChar.get(sid,H.SCHOOL_LABEL) for sid in sids}
    
    geoColor={schoolChar.get(sid,H.GEO):'k' for sid in sids}
    #geoSize={schoolChar.get(sid,H.GEO):np.sqrt(schoolValues[sid])*mult for sid in sids}
    geoSize={schoolChar.get(sid,H.GEO):(schoolValues[sid])*mult for sid in sids}
    
    ans=v._geoUtil.geoPartialScatter(ax,geoColor,geoSize)
    v._geoUtil.geoLabel(ax, labels,color='k',size='small')
    
    if outFile:
        _prepDir(outFile)
        plt.savefig(fname=outFile,bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    

def plotSupplyDemand(outFile,mult=1):
    ''' Plot the expected number of students from each geocode and the capacity of schools.'''
    field='K2 Capacity'
    schoolChar=SchoolSnapshot.loadCSV(PlanFiles.FIXED_EFFECT)
    capSchools=[sid for sid in schoolChar if schoolChar.get(sid,'Capacity School')=='1']
    v=PlanVisualizer()
    fig=plt.figure(figsize=(8,10))
    ax=plt.subplot(111)
    #v.prepareCanvas(ax)
    
    weights=data_reader.getDict(PlanFiles.GEO_CHAR, 1, True)
    for g in weights:
        weights[g]*=mult
    v._geoUtil.geoScatter(ax, {g:'b' for g in weights}, weights)
    
    H=ProgramCharHeader
    
    sids=[sid for sid in schoolChar if schoolChar.get(sid,field, 1)>0]
    labels={schoolChar.get(sid,H.GEO):schoolChar.get(sid,H.SCHOOL_LABEL) for sid in sids}
    
    value={sid:schoolChar.get(sid,field,1) for sid in sids}
    
       
    geoColor={schoolChar.get(sid,H.GEO):'y' for sid in sids}
    geoSize={schoolChar.get(sid,H.GEO):value[sid]*mult for sid in sids}
    
    v._geoUtil.geoPartialScatter(ax,geoColor,geoSize)
    
    geoColor={schoolChar.get(sid,H.GEO):'y' for sid in capSchools}
    geoSize={schoolChar.get(sid,H.GEO):value[sid]*mult for sid in capSchools}
    
    v._geoUtil.geoPartialScatter(ax,geoColor,geoSize,hatch=5*'+')
    
    v._geoUtil.geoLabel(ax, labels,color='k',size='x-small')
    _prepDir(outFile)
    plt.savefig(fname=outFile,bbox_inches='tight')



def plotQuality(inFile,distParam,outFile):
    ''' Plot the inferred quality of the schools'''
    #filename=os.path.join(PlanFolders.BASE,'fixed-effects-capacities.csv')
    schoolChar=SchoolSnapshot.loadCSV(inFile)
    v=PlanVisualizer()
    fig=plt.figure(figsize=(10,16))
    ax=plt.subplot(111)
    v.prepareCanvas(ax)
    H=ProgramCharHeader
    
    sids=[sid for sid in schoolChar if schoolChar.get(sid,'Quality', 0)]
    labels={schoolChar.get(sid,H.GEO):schoolChar.get(sid,H.SCHOOL_LABEL) for sid in sids}
    
    quality={sid:schoolChar.get(sid,'Quality',1)/distParam for sid in sids}
    minQ=0
    #minQ=min(quality.values())
    maxQ=2
    #maxQ=max(quality.values())
    minS=1
    maxS=100
    quality={sid:(quality[sid]-minQ)/maxQ*(maxS-minS)+minS for sid in sids}
    geoColor={schoolChar.get(sid,H.GEO):'k' for sid in sids}
    geoSize={schoolChar.get(sid,H.GEO):quality[sid] for sid in sids}
    
    ans=v._geoUtil.geoPartialScatter(ax,geoColor,geoSize)
    v._geoUtil.geoLabel(ax, labels,color='k',size='small')
    
    _prepDir(outFile)
    plt.savefig(fname=outFile,bbox_inches='tight')