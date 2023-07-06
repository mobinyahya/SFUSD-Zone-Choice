'''
Created on Jan 13, 2014

@author: pengshi
'''
import numpy as np
import random

class WeightedRandom(object):
    ''' A helper class for sampling objects from a list according to certain weights.'''
    @staticmethod
    def getFromWeights(obj,weights):
        if len(obj)!=len(weights):
            raise ValueError('Length of object is %d, not equal to length of weights %d'%(len(obj),len(weights)))
        w2=np.array(weights,dtype=float)
        w2=np.cumsum(w2/np.sum(w2))
        r=random.random()
        for i in range(len(w2)):
            if r<=w2[i]:
                break
        return obj[i]
        
                            