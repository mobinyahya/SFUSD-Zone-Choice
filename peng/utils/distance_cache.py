'''
Created on Jan 7, 2014

@author: pengshi
'''
from peng.utils.csv_char_data import SimpleCSVCharData


class DistanceCacheParams(object):
    ''' A class that maintains where the distance files are. 
    This is only needed if multiple distance files are present and one wishes to avoid conflicts.
    '''
    geoDistFile={}
    
    @staticmethod
    def update(year,geoDist):
        year=str(year)
        print('Updating distance %s to %s'%(year,geoDist))
        #DistanceCacheParams.bpsDistFile[year]=bpsDist
        DistanceCacheParams.geoDistFile[year]=geoDist

class DistanceCache(object):
    ''' A class used to read in and cache in memory the distances between neighborhoods (geocodes)
    and schools. '''
    
    MILE= 1609.34
    
    def __init__(self):
        self._cached={}
        
    def loadCache(self,year):
        print('Using distance cache files %s'%(DistanceCacheParams.geoDistFile[year]))
        year=str(year)
        geoDist=SimpleCSVCharData.loadCSV(DistanceCacheParams.geoDistFile[year],key='geocode')
        #bpsDist=SimpleCSVCharData.loadCSV(DistanceCacheParams.bpsDistFile[year],key='Student Code')
        self._cached[year]=geoDist
        return self
        
        
    def walkDist(self,year,geo,sid,realID=''):
        year=str(year)
        geo=str(geo)
        sid=str(sid)
        realID=str(realID)

        if year not in self._cached:
            self.loadCache(year)
            
        geoDist=self._cached[year]
            
        return float(geoDist.get(geo,sid))/self.MILE

if __name__=='__main__':
    dc=DistanceCache()
    #print dc.walkDist(2012,0,4260,207578)
    #print dc.walkDist(2013,0,4260,207578)
    #print dc.walkDist(2013,0,4080,207578)
    print(dc.walkDist(2013,3,4123,337734))
    
    