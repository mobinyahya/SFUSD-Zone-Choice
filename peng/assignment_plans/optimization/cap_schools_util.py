'''
Created on Feb 9, 2014

@author: pengshi
'''
import numpy as np

class CapSchoolsUtil(object):
    ''' A helper class manipulating data on the capacity school (default school)
    for each geocode.'''
    @staticmethod    
    def cleanCapSchools(logit,capSchools):
        for i in range(logit._T):
            if not i in capSchools:
                capSchools[i]=[]
            
    @staticmethod
    def makeApplyCap(logit,capSchools):
        ans=np.ones((logit._T,logit._S))
        for t in capSchools:
            for s in capSchools[t]:
                ans[t,s]=0
        return ans
    