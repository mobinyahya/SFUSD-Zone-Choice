'''
Created on Dec 28, 2012

Miscellaneous helper functions.

@author: pengshi
'''
import os

def _getArg(kwargs,name,default,noneSet=[],deleteEntry=False):
    ans= kwargs[name] if name in kwargs and (kwargs[name] not in noneSet) else default
    if deleteEntry and name in kwargs:
        del kwargs[name]
    return ans

def _extractNameFromPath(filename):
    head,tail=os.path.split(filename)
    return os.path.splitext(tail)[0]

def _stringToFilename(s):
    return s.strip().replace(' ','_').split(os.path.sep)[0]

def _getFileShortName(filename):
    return os.path.splitext(os.path.split(filename)[1])[0]

def _prepDir(file):
    head,tail=os.path.split(file)
    if not os.path.exists(head):
        os.makedirs(head)
        
def _isNumber(e):
    try:
        float(e)
        return True
    except:
        return False
    