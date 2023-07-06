'''
Created on Dec 26, 2012

@author: pengshi
'''
import abc

class Bin(object, metaclass=abc.ABCMeta):
    ''' An abstract class representing an item in the DA algorithm'''
    @abc.abstractproperty
    def capacity(self):
        pass
    
class DASTB_Bin(Bin, metaclass=abc.ABCMeta):
    ''' Represent an item in the DA algorithm with single tie-breakers.'''
    @abc.abstractproperty
    def program(self):
        pass
    
    @abc.abstractmethod
    def scoreBoost(self,student):
        pass
    
    def accessProb(self,student,cutoff):
        if self.capacity>0:
            return min(1,max(0,1+self.scoreBoost(student)-cutoff))
        else:
            return 0
    