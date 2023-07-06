'''
Created on Jul 1, 2019

This script draws all of the figures related to the empirical application in the paper.

@author: pengshi
'''
from peng.assignment_plans.plotting.utils import plotSupplyDemand, plotQuality, plotField, plotQuotas, plotSchoolCost
from peng.constants.locations import PlanFolders, Folders
import os


if __name__ == '__main__':
    plotSupplyDemand(os.path.join(PlanFolders.RESULT_FIGURES,'D-1a.pdf'),2)
    plotQuality(os.path.join(Folders.INPUTS,'school_characteristics.csv'), 0.531,os.path.join(PlanFolders.RESULT_FIGURES,'D-1b.pdf'))
    
    name='OptimizedPlan'
    plotField(name,'Utility',3,20,os.path.join(PlanFolders.RESULT_FIGURES,'F-1a.pdf'))
    plotField(f'{name}-updated','Utility',3,20,os.path.join(PlanFolders.RESULT_FIGURES,'F-1b.pdf'))
    
    plotQuality(os.path.join(Folders.INPUTS,'school_characteristics.csv'), 0.531,os.path.join(PlanFolders.RESULT_FIGURES,'F-2a.pdf'))
    plotQuality(os.path.join(Folders.INPUTS,'school_characteristics_2014.csv'), 0.610, os.path.join(PlanFolders.RESULT_FIGURES,'F-2b.pdf'))
    
    
    
    mult1=10
    mult2=10
    plotSchoolCost(name,mult1,os.path.join(PlanFolders.RESULT_FIGURES,'4a.pdf'))
    plotQuotas(name,mult1,mult2, os.path.join(PlanFolders.RESULT_FIGURES,'4b.pdf'),os.path.join(PlanFolders.RESULT_FIGURES,'4c.pdf'))
    