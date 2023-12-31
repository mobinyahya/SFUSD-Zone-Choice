'''
Created on Feb 6, 2014

@author: pengshi
'''
import os
from peng.constants.locations import PlanFolders
from peng.utils import data_reader, data_saver
from collections import OrderedDict

class ConvertPrioritiesParams(object):
    '''A helper class that changes the plan generated by the optimization engine
    into the format used by the simulation engine'''
    def process(self):
        for fname in os.listdir(PlanFolders.PRIORITIES):
            if fname[-4:]=='.csv':
                inFile=os.path.join(PlanFolders.PRIORITIES,fname)
                outFile=os.path.join(PlanFolders.PARAMS,fname.replace('.csv','Params.csv'))
                self.convert(outFile,inFile)
            
    def convert(self,outFile,inFile):
        orig=data_reader.getDict(inFile,2,True)
        ans=OrderedDict()
        for geo in orig:
            ans[geo]=OrderedDict()
            for s in orig[geo]:
                if orig[geo][s]>0:
                    ans[geo][s]={'Priority':orig[geo][s]}
        data_saver.saveDict(outFile,ans,3,['Geocode','School Code'])
                
        
        
if __name__=='__main__':
    ConvertPrioritiesParams().process()