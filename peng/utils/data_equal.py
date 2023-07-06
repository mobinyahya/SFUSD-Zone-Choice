'''
Created on Dec 21, 2012

Functions that are used to check if two classes contain the same data.
This is only used for debugging, and is not necessary in normal use.

@author: pengshi
'''


from abc import abstractmethod, ABCMeta
import numpy as np

class DataEqualable(metaclass=ABCMeta):
    @abstractmethod
    def dataEquals(self,object):
        pass

def equal(object1,object2):
    if np.all(object1==object2):
        return True
    elif isinstance(object1,DataEqualable) and isinstance(object2,DataEqualable):
        return object1.dataEquals(object2)
    elif isSequence(object1) and isSequence(object2):
        return seqEqual(object1,object2)
    elif isMap(object1) and isMap(object2):
        return mapEqual(object1,object2)
    else:
        return False

def hasDataEquals(object):
    return hasattr(object,'dataEquals') and callable(object.dataEquals)
    
def isSequence(object):
    return type(object)!=str and hasattr(object,'__getitem__') and hasattr(object,'__len__') and not (hasattr(object,'keys') and callable(object.keys))

def isMap(object):
    return hasattr(object,'__getitem__') and hasattr(object,'__len__') and hasattr(object,'__contains__') and (hasattr(object,'keys') and callable(object.keys))

def seqEqual(seq1,seq2):
    if len(seq1)!=len(seq2):
        return False
    for i in range(len(seq1)):
        if not equal(seq1[i],seq2[i]):
            return False
    return True

def mapEqual(map1,map2):
    if len(map1)!=len(map2):
        return False
    for k in list(map1.keys()):
        if not k in map2:
            return False
        if not equal(map1[k], map2[k]):
            return False
    return True