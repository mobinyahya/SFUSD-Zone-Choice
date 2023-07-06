'''
Created on Nov 24, 2012

@author: pengshi
'''
import csv
from peng.utils import data_reader
from peng.constants.locations import Files

def lazy_property(fn):
    attr_name = '_lazy_' + fn.__name__
    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazyprop

class DataLoader(object):
    ''' A class for loading certain data files and storing it in memory. '''
    @lazy_property
    def googleDist(self):
        return data_reader.getGoogleDist(Files.GOOGLE_GEO_DIST,inMiles=True)

    @lazy_property 
    def walkGeo(self):
        return data_reader.getWalkGeo(Files.WALK_ZONE)
    
    
