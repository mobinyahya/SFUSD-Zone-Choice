'''
Created on Jan 14, 2013

@author: pengshi
'''
from peng.utils.geo_util import GeoUtil
import colorsys
import random
from peng.deferred_acceptance.assignment import Assignment
from peng.utils import data_reader
import matplotlib.pyplot as plt
from peng.utils.common import _getArg
from matplotlib.font_manager import FontProperties
from collections import Counter
from peng.utils.program import ProgramCharHeader


class PlanVisualizer(object):
    ''' A class containing functionality for displaying the choicesets of various neighborhoods'''
    def __init__(self):
        self._geoUtil=GeoUtil()
        
    def prepareCanvas(self,ax,geos=[],**kwargs):
        if not geos:
            geos=[str(i) for i in range(868)]
        geoColor={g:'k' for g in geos}
        geoSize={g:1 for g in geos}
        return self._geoUtil.geoScatter(ax, geoColor, geoSize,**kwargs)
    
    def colorArea(self,ax,geos,color,size,**kwargs):
        geoColor={g:color for g in geos}
        geoSize={g:size for g in geos}
        return self._geoUtil.geoPartialScatter(ax, geoColor, geoSize,**kwargs)
    
    def drawStudents(self,ax,sset,color='k',mult=4,**kwargs):
        geoSize=Counter()
        for s in sset:
            geoSize[s.geo]+=mult
        geoColor={g:color for g in geoSize}
        ret=self._geoUtil.geoPartialScatter(ax, geoColor, geoSize,**kwargs)
        return ret
        
    
    def drawSchoolTypes(self,ax,schoolChar,headers,ranges,**kwargs):
        names=_getArg(kwargs,'names',headers)
        colors=_getArg(kwargs,'colors',['k']*len(headers))
        print(colors)
        markers=_getArg(kwargs,'markers',['s']*len(headers))
        ret=[]
        for i,h in enumerate(headers):
            sids=[sid for sid in schoolChar if schoolChar.get(sid,h,True) in ranges[i]]
            ret.append(self.drawSchools(ax,schoolChar,sids,marker=markers[i],color=colors[i],labelColor='k'))
        plt.legend(ret,names,loc='lower right',bbox_to_anchor=(1,0),scatterpoints=1)
        
    def drawSchools(self,ax,schoolChar,sids,color='k',labelColor='k',size=20,**kwargs):
        H=ProgramCharHeader
        geoColor={schoolChar.get(sid,H.GEO):color for sid in sids}
        geoSize={schoolChar.get(sid,H.GEO):size for sid in sids}
        
        labels={schoolChar.get(sid,H.GEO):schoolChar.get(sid,H.SCHOOL_LABEL) for sid in sids}
        ans=self._geoUtil.geoPartialScatter(ax,geoColor,geoSize,**kwargs)
        self._geoUtil.geoLabel(ax, labels,color=labelColor,size='x-small')
        return ans
    
    def drawProgram(self,ax,p,color='k',size=20,labelAppend='',**kwargs):
        g=p.char[ProgramCharHeader.GEO]
        geoColor={g:color}
        geoSize={g:size}
        labels={g:'%s %s%s'%(p.char[ProgramCharHeader.SCHOOL_LABEL],p.id.pid,labelAppend)}
        ans=self._geoUtil.geoPartialScatter(ax, geoColor, geoSize,**kwargs)
        self._geoUtil.geoLabel(ax, labels,size='x-small')
        return ans
    
    def drawMenu(self,ax,menu,student,**kwargs):
        studentColor=_getArg(kwargs,'studentColor','b')
        studentSize=_getArg(kwargs,'studentSize',20)
        studentMarker=_getArg(kwargs,'studentMarker','o')
        studentLabel=_getArg(kwargs,'studentLabel','%s-%s'%(student.grade,student.kind))
        schoolColor=_getArg(kwargs,'schoolColor',studentColor)
        schoolSize=_getArg(kwargs,'schoolSize',20)
        schoolMarker=_getArg(kwargs,'schoolMarker','s')
        ret1=self._geoUtil.geoPartialScatter(ax,{student.geo:studentColor},{student.geo:studentSize},marker=studentMarker)
        
        ps=menu.getOptions(student)
        geos={}
        schoolProgs={}
        H=ProgramCharHeader
        for p in ps:
            sl=p.char[H.SCHOOL_LABEL]
            pid=p.id.pid
            if not sl in schoolProgs:
                schoolProgs[sl]=[]
            if not pid in schoolProgs[sl]:
                schoolProgs[sl].append(pid)
            g=p.char[H.GEO]
            if not g in geos:
                geos[g]=set()
            geos[g].add(sl)
            
        geoColor={}
        geoSize={}
        labels={}
        for g in geos:
            labels[g]='\n'.join(['%s-%s'%(sl,','.join(schoolProgs[sl])) for sl in geos[g]])
            geoColor[g]=schoolColor
            geoSize[g]=schoolSize
        ret2=self._geoUtil.geoPartialScatter(ax, geoColor, geoSize,marker=schoolMarker)
        self._geoUtil.geoLabel(ax, labels,size='xx-small')
        return [ret1,ret2]
    
    def drawZones(self,ax,zoneFile,**kwargs):
        intZone=_getArg(kwargs,'intZone',True)
        random.seed(0)
        geoZone=Assignment(data_reader.getDict(zoneFile,1,False),none='')
        zones=sorted(geoZone.values())
        if intZone:
            zonesInt=sorted([int(z) for z in zones])
            zones=[str(z) for z in zonesInt]
        
        if 'zColor' not in kwargs:
            zColor={}
            numZones=len(zones)
            numDiff=_getArg(kwargs,'numDiff',1)
            for i,z in enumerate(zones):
                zColor[z]=colorsys.hsv_to_rgb(((i*numDiff)%numZones)/float(numZones), 1, 1)
        else:
            zColor=kwargs['zColor']
        ret=[]
        labels=[]
        for z in zones:
            ret.append(self._geoUtil.geoPartialScatter(ax, {g:zColor[z] for g in geoZone.assignedTo(z)}))
            if intZone:
                labels.append('Zone %s'%z)
            else:
                labels.append(z)
        legend_font_props = FontProperties()
        legend_font_props.set_size(6)
        plt.legend(ret,labels,loc='lower right',bbox_to_anchor=(1,0),ncol=1,scatterpoints=1,markerscale=2,prop=legend_font_props)
        