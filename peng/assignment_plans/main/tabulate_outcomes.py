'''
Created on Jul 1, 2019

This script performs all the simulations and create all of the computational tables in the paper.

@author: pengshi
'''
from peng.constants.locations import PlanFolders, Folders
import os
from peng.assignment_plans.simulations.finite_market_simulator import FiniteMarketSimulator
from peng.assignment_plans.generic.random_menu_plan import SimpleRandomMenuPlan
from peng.assignment_plans.simulations.evaluate_as_large import EvaluateAsLarge

def mainEvaluations(n):
    resFile=os.path.join(PlanFolders.RESULT_TABLES,'Table1.csv')
    use2014=False
    for name in ['3Zone','HomeBased','AshlagiShi2015','OptimizedPlan']:
        if name=='3Zone':
            capSchoolFile=os.path.join(Folders.INPUTS,'capacity_schools_3zone.csv')
            oldWalkPrio=True
        else:
            capSchoolFile=''
            oldWalkPrio=False
        sim=FiniteMarketSimulator(use2014,capSchoolFile=capSchoolFile)
        paramFile=os.path.join(PlanFolders.PARAMS,'%sParams.csv'%name)
        sim.simulate(name, paramFile, n,oldWalkPrio=oldWalkPrio)
        sim.analyze(resFile, name,new=False)
        
def updatedEvaluations(n):
    resFile=os.path.join(PlanFolders.RESULT_TABLES,'Table4.csv')
    use2014=True
    for name in ['3Zone','HomeBased','OptimizedPlan']:
        if name=='3Zone':
            capSchoolFile=os.path.join(Folders.INPUTS,'capacity_schools_3zone.csv')
            oldWalkPrio=True
        else:
            capSchoolFile=''
            oldWalkPrio=False
        sim=FiniteMarketSimulator(use2014,capSchoolFile=capSchoolFile)
        paramFile=os.path.join(PlanFolders.PARAMS,'%sParams.csv'%name)
        simName=f'{name}-updated'
        sim.simulate(simName, paramFile, n,oldWalkPrio=oldWalkPrio)
        sim.analyze(resFile, simName,new=False)
        
def checkConvergence(n):
    name='OptimizedPlan'    
    eal=EvaluateAsLarge().evaluate(os.path.join(PlanFolders.RESULT_TABLES,'Table5-column1.csv'), [name])
    
    resFile=os.path.join(PlanFolders.RESULT_TABLES,'Table5-column2.csv')
    plan=SimpleRandomMenuPlan.planFromName(name)
    capFile=os.path.join(Folders.LOGS,'%s_ideal_capacities.csv'%name)
    plan.saveExpectedCapacities(capFile)
    sim=FiniteMarketSimulator(use2014Params=False,mult=1,totalVariation=False,regionalVariation=False,useCapFile=capFile)
    paramFile=os.path.join(PlanFolders.PARAMS,'%sParams.csv'%name)
    sim.simulate(name, paramFile, n)
    sim.analyze(resFile, name,new=False)    
        

if __name__=='__main__':
    n=100000
    mainEvaluations(n)
    updatedEvaluations(n)
    checkConvergence(n)
    
