'''
Created on Dec 24, 2012

@author: pengshi
'''
import abc
from abc import abstractproperty

class Chooser(object, metaclass=abc.ABCMeta):
    '''An abstract class representing an agent in the DA algorithm.'''
    @abstractproperty
    def choices(self):
        return
    
    def choiceNum(self,option):
        for i,c in enumerate(self.choices):
            if option==c:
                return i+1
        return None
    
    def _choicesStr(self):
        rows=['Choices:']
        for i,c in enumerate(self.choices):
            rows.append('%d : %s'%(i+1,str(c)))
        return '\t'.join(rows)
    
class SimpleChooser(Chooser):
    def __init__(self,identity,choices):
        self._identity=identity
        self._choices=choices
        
    @property
    def choices(self):
        return self._choices
    
    @property
    def identity(self):
        return self._identity
    
class SimpleTypeChooser(Chooser):
    ''' A representation of an agent in the DA algorithm.'''
    def __init__(self,identity,Type,choices):
        self._identity=identity
        self._type=Type
        self._choices=choices
        
    @property
    def choices(self):
        return self._choices
    
    @property
    def identity(self):
        return self._identity
    
    @property
    def type(self):
        return self._type
    
    def __repr__(self):
        return f'Chooser {self.identity} of type {self.type} with choices {[b.identity for b in self.choices]}'
    

