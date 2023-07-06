'''
Created on Jan 1, 2013

@author: pengshi
'''
from peng.utils.char_reader import CharFileReader
from peng.constants.headers import GeocodeCoords
import matplotlib.pyplot as plt

import numpy as np
from peng.utils.common import _getArg
from peng.constants.locations import Files
from collections import Counter
import os
from peng.utils.GeoCoordUtil import GeoCoordUtil
from peng.utils.data_cache import DataLoader

class GeoUtil(object):
    ''' A helper class with functionality on processing longitude and latitude data.'''
    def __init__(self,filename=Files.GEOCODE_COORDS,**kwargs):
        self._geos,self._long,self._lat,self._area=self._readData(filename)
        self._geoSet=set(self._geos)
        
        
    def _readData(self,filename):
        gs=set()
        longs={}
        lats={}
        areas={}
        H=GeocodeCoords
        for line in CharFileReader(filename):
            g=line[H.GEO].strip()
            longs[g]=float(line[H.LONG])
            lats[g]=float(line[H.LAT])
            areas[g]=float(line[H.AREA])*6.8404970e-8
            gs.add(g)
        return gs,longs,lats,areas
    
    def long(self,geo):
        if not str(geo) in self._long:
            return self.long(str(int(geo)-1))
        return self._long[str(geo)]
    
    def lat(self,geo):
        if not str(geo) in self._lat:
            return self.lat(str(int(geo)-1))
        return self._lat[str(geo)]
    
    def area(self,geo):
        geo=str(geo)
        if not geo in self._area:
            return self.area(str(int(geo)-1))
        return self._area[geo]
            
    @property
    def geos(self):
        return self._geos
    
    def isValidGeo(self,geo):
        return str(geo) in self._geoSet
        
    
    def transformedLong(self,geo):
        conv=np.pi/2/90
        return np.cos(self.lat(geo)*conv)*self.long(geo)
    
    def straightDist(self,geo1,geo2):
        geo1=str(geo1)
        geo2=str(geo2)
        lat1=self.lat(geo1)
        lon1=self.long(geo1)
        lat2=self.lat(geo2)
        lon2=self.long(geo2)
        return GeoCoordUtil.dist(lon1, lat1, lon2, lat2)
    
    def walkDist(self,geo1,geo2):
        if not hasattr(self,'_googleDist'):
            self._googleDist=DataLoader().googleDist
        if not geo1 in self._googleDist:
            return self.walkDist(str(int(geo1)-1), geo2)
        if not geo2 in self._googleDist:
            return self.walkDist(geo1,str(int(geo2)-1))
        return self._googleDist[geo1][geo2]
    
    def _getBounds(self):
        x=[self.long(g) for g in self.geos]
        y=[self.lat(g) for g in self.geos]
        return max(x),min(x),max(y),min(y)
    
    def geoLabel(self,ax,geoLabels,**kwargs):
        for g in geoLabels:
            x=self.long(g)
            y=self.lat(g)
            ax.annotate(geoLabels[g],xy=(x,y),xytext=(0,-10),textcoords='offset points',**kwargs)


    def geoPartialScatter(self,ax,geoColor,geoSize={},**kwargs):
        geos=sorted(self.geos.intersection(set(geoColor.keys())))
        n=len(geos)
        x=np.zeros(n)
        y=np.zeros(n)
        s=np.zeros(n)
        c=[]
        for i,g in enumerate(geos):
            #x[i]=self.transformedLong(g)
            x[i]=self.long(g)
            y[i]=self.lat(g)
            s[i]=20 if g not in geoSize else geoSize[g]
            c.append(geoColor[g])
        return ax.scatter(x,y,s,c,**kwargs)
    
    def geoScatter(self,ax,geoColor,geoSize={},**kwargs):
        self.prepareCanvas(ax)
        return self.geoPartialScatter(ax, geoColor, geoSize,**kwargs)
    
    def prepareCanvas(self,ax):
        ax.get_xaxis().set_visible(False)
        maxx,minx,maxy,miny=self._getBounds()
        diffx=maxx-minx
        diffy=maxy-miny
        marg=0.025
        ax.set_xlim((minx-marg*diffx,maxx+marg*diffx))
        ax.set_ylim((miny-marg*diffy,maxy+marg*diffy))
        ax.get_yaxis().set_visible(False)
    
    
    def plotGeoDict(self,input,newMin=0,mult=1,outFile=''):
        values=Counter()
        for geo in input:
            values[geo]=(input[geo]-newMin)*mult+1
            #values[geo]=np.sqrt(input[geo]-newMin)*mult
        fig=plt.figure(figsize=(12,16))
        ax=plt.subplot(111)
        self.prepareCanvas(ax)
        geoUtil=GeoUtil()
        geoUtil.geoPartialScatter(ax, {geo:'k' for geo in values}, geoSize=values)
        if not outFile:
            plt.show()
        else:
            base,ext=os.path.split(outFile)
            if not os.path.exists(base):
                os.makedirs(base)
            
            plt.savefig(fname=outFile,bbox_inches='tight')
            plt.close()
        