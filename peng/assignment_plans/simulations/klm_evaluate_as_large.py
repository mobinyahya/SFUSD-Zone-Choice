'''
Created on Feb 6, 2014

@author: pengshi
'''
import numpy as np
import pandas as pd
from peng.utils import data_reader
from peng.constants.locations import PlanFolders, PlanFiles
import os
from peng.assignment_plans.generic.fixed_effect_utility import FixedEffectUtility
from peng.utils.csv_char_data import SimpleCSVCharData
from peng.assignment_plans.generic.random_menu_plan import SimpleRandomMenuPlan
from peng.utils.distance_cache import DistanceCache


class EvaluateAsLarge(object):
    ''' This class is used to evaluate a certain budget set probability matrix in the continuum model.
    It is tailored to the Boston application.'''

    def __init__(self):
        self.utilParam = FixedEffectUtility.sfusd2018K()
        weights = pd.read_csv(PlanFiles.STUDENT_COUNTS, index_col=0)
        self.weightDict = dict(zip(weights.index, weights.student_count))
        # self.weightDict = data_reader.getDict(PlanFiles.GEO_CHAR, 1, True)
        costDict = pd.read_csv(PlanFiles.DISTANCES, index_col=[0, 1]).values
        self.costDict = np.hstack([costDict, np.zeros((costDict.shape[0], 1))])
        # self.costDict = data_reader.getDict(PlanFiles.MILES_BUSED, 2, True)

    def inferChoiceSetDict(self, cutoffs, weightDict):
        ans = {}
        for i in weightDict:
            ans[i] = set([j for j in cutoffs[i] if cutoffs[i][j] > 0])
            if len(ans[i]) == 0:
                raise ValueError('Choice menu of geocode %s is empty' % i)
        return ans

    def evaluate(self, outFile, nameList):
        char = SimpleCSVCharData()
        convertCutoffs = False
        for name in nameList:
            print('Evaluating Plan %s' % name)
            plan = SimpleRandomMenuPlan.planFromName(name, convertCutoffs)
            # name+='-cutoffs' if convertCutoffs else ''
            # dc = DistanceCache()
            num = 0
            denum = 0
            for geo in plan.iset:
                for s in plan.choiceSet(geo):
                    # num += plan.assignmentProbability(geo, s) * dc.walkDist('2014', geo, s) * plan.weight(geo)
                    num += plan.assignmentProbability(geo, s) * self.costDict[int(geo), int(s)] * plan.weight(geo)
                denum += plan.weight(geo)
            char.set('(1) Av. # of choices', name, plan.averageNumChoices())
            char.set('(2) Av. miles to assigned school', name, float(num) / denum)

            char.set('(3) Miles bused per student', name, plan.averageCost())
            # char.set('(4) Av. bus coverage area', name, plan.averageBusingArea())
            # char.set('(5) Av. # of busing choices', name, plan.averageBusingChoices())

            char.set('(6) Weighted average utility', name, plan.averageUtilityCondAssigned())
            char.set('(7) 10th percentile utility', name, plan.utilityQuantile(10))
            char.set('(8) Lowest utility of any neighborhood', name, plan.minimumUtilityCondAssigned())
            char.set('(8) Lowest utility neighborhood', name, plan.minimumUtilityTypeCondAssigned())

            char.set('(9) % getting top 1 choice in menu', name, plan.averageFirstChoiceProb())
            char.set('(10) % getting top 3 choice in menu', name, plan.averageTop3ChoiceProb())

        char.saveCSV(outFile)


