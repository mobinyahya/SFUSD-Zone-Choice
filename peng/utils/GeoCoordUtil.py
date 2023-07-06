'''
Created on Dec 6, 2013

@author: pengshi
'''
import numpy as np

class GeoCoordUtil(object):
    ''' A helper class used by the GeoUtil class'''
    RADIUS=3963.1 # radius of earth in miles
    
    @staticmethod
    def toXYZ(lon,lat):
        lon=float(lon)
        lat=float(lat)
        z=GeoCoordUtil.RADIUS*np.sin(np.pi*lat/180)
        xy=GeoCoordUtil.RADIUS*np.cos(np.pi*lat/180)
        x=xy*np.cos(np.pi*lon/180)
        y=xy*np.sin(np.pi*lon/180)
        return np.array([x,y,z])
    
    @staticmethod
    def dist(lon1,lat1,lon2,lat2):
        c1=GeoCoordUtil.toXYZ(lon1, lat1)
        c2=GeoCoordUtil.toXYZ(lon2, lat2)
        diff=c1-c2
        return np.sqrt(np.sum(diff*diff))
    
    
    