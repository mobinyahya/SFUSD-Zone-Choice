'''
Created on Jun 28, 2019

This script computes the optimized plan for Boston as well as the theoretical upper bound.

@author: pengshi
'''

from peng.assignment_plans.optimization.optimized_plan import solveCertain
# from peng.assignment_plans.simulations.evaluate_as_large import EvaluateAsLarge
from peng.assignment_plans.simulations.klm_evaluate_as_large import EvaluateAsLarge
import os
from peng.constants.locations import PlanFolders

if __name__ == '__main__':
    
    # solveCertain('OptimizedPlan',0.6,0.5,8.5,6.2,True)
    #
    # solveCertain('UpperBound',0.64,0.5,8.52,8.18,False)
    # EvaluateAsLarge().evaluate(os.path.join(PlanFolders.RESULT_OTHER, 'theoretical_upper_bound.csv'), ['UpperBound'])
    EvaluateAsLarge().evaluate(os.path.join(PlanFolders.RESULT_OTHER, 'theoretical_upper_bound.csv'), ['dist10_maxfrl1_alpha0.5_card0_umodelavg'])
    # EvaluateAsLarge().evaluate(os.path.join(PlanFolders.RESULT_OTHER, 'theoretical_upper_bound.csv'), ['dist4_maxfrl1_alpha1_card0_umodelavg', 'dist4_maxfrl0.9_alpha1_card0_umodelavg', 'dist4_maxfrl0.9_alpha1_card8_umodelavg', 'dist4_maxfrl1_alpha0.5_card0_umodelavg'])