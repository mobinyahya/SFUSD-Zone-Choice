"""
Created on Jun 27, 2019

@author: pengshi
"""
import pandas as pd

from peng.assignment_plans.optimization.primal_solver import PrimalSolver
from peng.assignment_plans.optimization.dual_solver import DualSolver
from peng.utils.common import _getArg
from peng.assignment_plans.optimization.fast_logit import FastLogit
import numpy as np
from peng.assignment_plans.generic.fixed_effect_utility import FixedEffectUtility
from peng.utils import data_saver  # data_reader
import os
from peng.constants.locations import PlanFolders, PlanFiles  # , Folders

# from peng.utils.school_snapshot import SchoolSnapshot
from peng.utils.csv_char_data import SimpleCSVCharData
from peng.assignment_plans.generic.random_menu_plan import RandomMenuPlan
from peng.assignment_plans.optimization.convert_priorities_params import (
    ConvertPrioritiesParams,
)

from peng.assignment_plans.optimization.primal_solver import PrimalSolver


class OptimizedPlan(RandomMenuPlan):
    """Class that computes the optimal assignment plan for Boston. It handles all the data preparation, and then
    uses the PrimalSolver class to do the computations. 
    """

    def __init__(
        self,
        weightDict,
        utilParam,
        capacities,
        distance,
        averageBudget,
        frl,
        max_frl,
        sibling_frl=None,
        original_capacity=None,
        **kwargs
    ):

        self._utilParam = utilParam
        alpha = _getArg(kwargs, "alpha", 1)
        # sizeBudget = _getArg(kwargs, "sizeBudget", 0)

        self._name = _getArg(
            kwargs,
            "name",
            "Optimal Plan with average budget %s and alpha %s" % (averageBudget, alpha),
        )
        # reOptimize = _getArg(kwargs, "reOptimize", True)
        # capSchool = _getArg(
        #     kwargs, "capSchool", data_reader.getDict(PlanFiles.CAP_SCHOOLS, 1, False)
        # )

        totalWeight = sum(weightDict[i] for i in utilParam.iset if i in weightDict)
        budget = averageBudget * totalWeight
        cardBudget = _getArg(kwargs, "cardBudget", 0) * totalWeight

        self._k = _getArg(kwargs, "k", 0)

        self._iset = set(i for i in utilParam.iset if i in weightDict)

        (
            self._choiceSet,
            self._randomMenuDict,
            self._quota,
            self._numAllowed,
            self._lamda,
            self._gamma,
            self._xi,
        ) = self._solve(
            utilParam,
            budget,
            # sizeBudget,
            cardBudget,
            capacities,
            weightDict,
            distance,
            frl,
            max_frl,
            # capSchool,
            alpha=alpha,
            # reOptimize,
            original_capacity=original_capacity,
            sibling_frl=sibling_frl,
        )

        self._weightDict = {}
        self._costDict = {}
        for i in self._iset:
            self._weightDict[i] = weightDict[i]
            self._costDict[i] = {}
            for j in self._choiceSet[i]:
                if j in distance[i]:
                    self._costDict[i][j] = distance[i][j]
                else:
                    self._costDict[i][j] = 0

    @staticmethod
    def sfusd2018K(
        averageBudget,
        max_frl,
        alpha=1,
        sizeBudget=0,
        cardBudget=0,
        k=0,
        reOptimize=False,
    ):
        weights = pd.read_csv(PlanFiles.STUDENT_COUNTS, index_col=0)
        weight_dict = dict(zip(weights.index, weights.student_count))
        util_param = FixedEffectUtility.sfusd2018K()
        capacities = pd.read_csv(PlanFiles.CAPACITIES)["remaining_cap"].to_numpy()
        distance = pd.read_csv(PlanFiles.DISTANCES, index_col=[0, 1]).values
        frl = pd.read_csv(PlanFiles.FRL)[["frl"]].to_numpy()
        tmp =pd.read_csv(PlanFiles.SIBLING_FRL)
        sibling_frl = tmp.frl.to_numpy()
        original_capacity = tmp.capacity.to_numpy()
        return OptimizedPlan(
            weight_dict,
            util_param,
            capacities,
            distance,
            averageBudget,
            frl,
            max_frl,
            alpha=alpha,
            # sizeBudget=sizeBudget,
            cardBudget=cardBudget,
            k=k,
            reOptimize=reOptimize,
            sibling_frl=sibling_frl,
            original_capacity=original_capacity,
        )

    # @staticmethod
    # def default2013K2(averageBudget,alpha=1,sizeBudget=0,cardBudget=0,k=0,reOptimize=True):
    #     geoChar=SimpleCSVCharData.loadCSV(PlanFiles.GEO_CHAR,key='Geocode')
    #     areaDict=geoChar.getColumn('Area', True)
    #     weightDict=geoChar.getColumn('Weight', True)
    #     utilParam=FixedEffectUtility.simple2013K12()
    #     initCostDict=data_reader.getDict(PlanFiles.MILES_BUSED,2,True)
    #     charFile=PlanFiles.FIXED_EFFECT
    #     capacities=SchoolSnapshot.loadCSV(charFile).getColumn('K2 Capacity', True)
    #     return OptimizedPlan(weightDict,areaDict,utilParam,initCostDict,capacities,averageBudget,alpha=alpha,sizeBudget=sizeBudget,cardBudget=cardBudget,k=k,reOptimize=reOptimize)
    #
    # @staticmethod
    # def default2013K2Small(averageBudget,alpha=1,sizeBudget=0,cardBudget=0,k=0,reOptimize=True):
    #     geoChar=SimpleCSVCharData.loadCSV(PlanFiles.GEO_CHAR,key='Geocode')
    #     areaDict=geoChar.getColumn('Area', True)
    #     weightDict=geoChar.getColumn('Weight', True)
    #     utilParam=FixedEffectUtility.simple2013K12()
    #     initCostDict=data_reader.getDict(PlanFiles.MILES_BUSED,2,True)
    #     charFile=os.path.join(Folders.INPUTS,'testing_school_characteristics.csv')
    #     capacities=SchoolSnapshot.loadCSV(charFile).getColumn('K2 Capacity', True)
    #     return OptimizedPlan(weightDict,areaDict,utilParam,initCostDict,capacities,averageBudget,alpha=alpha,sizeBudget=sizeBudget,cardBudget=cardBudget,k=k,reOptimize=reOptimize)

    def quota(self, i):
        return self._quota[i]

    def lamda(self, j):
        return self._lamda[j]

    @property
    def gamma(self, j):
        return self._gamma

    def saveQuota(self, outFile):
        header = ["Type", "Probability", "Quota", "numAllowed"]
        lines = []
        for i in self.iset:
            for k, (M, prob) in enumerate(self.randomMenus(i)):
                lines.append([i, prob, self._quota[i][k], self._numAllowed[i][k]])
        data_saver.saveLines(outFile, lines, header)

    def saveShadowPrice(self, outFile):
        data = SimpleCSVCharData()
        # data.set("BusingArea", "ShadowPrice", self._xi[0])
        # data.set("BusingCardinality", "ShadowPrice", self._xi[1])
        data.set("BusingCardinality", "ShadowPrice", self._xi)
        data.set("BusingDistance", "ShadowPrice", self._gamma)
        for j in self._lamda:
            data.set(j, "ShadowPrice", self._lamda[j])
        data.saveCSV(outFile)

    @property
    def utilParam(self):
        return self._utilParam

    def cost(self, i, j):
        return self._costDict[i][j]

    @property
    def randomMenuDict(self):
        return self._randomMenuDict

    @property
    def iset(self):
        return self._iset

    def choiceSet(self, i):
        return self._choiceSet[i]

    def weight(self, i):
        return self._weightDict[i]

    @property
    def name(self):
        return self._name

    def _solve(
        self,
        utilParam,
        budget,
        # sizeBudget,
        cardBudget,
        capacities,
        weightDict,
        # capSchool,
        distances,
        frl,
        max_frl,
        alpha,
        # reOptimize,
        sibling_frl=None,
        original_capacity=None
    ):
        logit = FastLogit.fromUtilParam(utilParam)
        c = np.zeros((logit.T, logit.S))
        B = budget
        # C = sizeBudget * logit.S
        D = cardBudget
        q = capacities  # np.zeros(logit.S)  # capacities
        w = np.zeros(logit.T)  # weight, student count
        # s=np.zeros(logit.T)  # neighborhood area
        # for i in initCostDict:
        #     if i in logit.iset:
        #         for j in initCostDict[i]:
        #             if j in logit.jset:
        #                 c[logit.iind(i)][logit.jind(j)]=initCostDict[i][j]
        for i in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                c[logit.iind(i)][logit.jind(j)] = distances[i, j]
        # capSchools = {logit.iind(i): [] for i in logit.iset}
        # for i in logit.iset:
        #     capSchools[logit.iind(i)].append(logit.jind(capSchool[i]))
        # print '%d-%d'%(logit.iind(i),logit.jind(j))

        for i in weightDict:
            if i in logit.iset:
                w[logit.iind(i)] = weightDict[i]
                # s[logit.iind(i)]=areaDict[i]
        # print('logit iset', logit.iset)
        # print("w", w)

        # for j in capacities:
        #     if j in logit.jset:
        #         q[logit.jind(j)] = capacities[j]

        primal = PrimalSolver(logit, c, B, D, q, w, frl, max_frl, alpha=alpha, original_capacity=original_capacity, sibling_frl=sibling_frl, k=self._k)
        # primal = DualSolver(logit, c, B, D, q, w, frl, max_frl, alpha=alpha, k=self._k)
        (
            z,
            Ms,
            rawQuota,
            rawNumAllowed,
            mu,
            nu,
            rawLamda,
            gamma,
            xi,
            frl_price,
        ) = primal.solve(True)
        # z,Ms,rawQuota,mu,nu,rawLamda,gamma=self._fakeSolve(logit.T,logit.S)

        # resolve using only schools in menus. Note that capschools are assumed to be always in menu.
        # if reOptimize:
        #     from itertools import chain
        #
        #     allowed = {t: set(chain.from_iterable(Ms[t])) for t in Ms}
        #     # primal=PrimalSolver2(logit,c,B,q,w,capSchools,alpha,allowed=allowed,k=logit.S+1)
        #     primal = PrimalSolver(
        #         logit, c, B, 0, 0, q, w, s, capSchools, alpha, allowed=allowed
        #     )
        #     z, Ms, rawQuota, rawNumAllowed, mu, nu, rawLamda, gamma, xi = primal.solve(
        #         True
        #     )

        choiceSet = {}
        randomMenuDict = {}
        quota = {}
        numAllowed = {}
        lamda = {}
        realJ = {logit.jind(j): j for j in logit.jset}
        for j in logit.jset:
            s = logit.jind(j)
            lamda[j] = rawLamda[s]
        for i in self.iset:
            choiceSet[i] = set()
            randomMenuDict[i] = []
            t = logit.iind(i)
            quota[i] = rawQuota[t]
            numAllowed[i] = rawNumAllowed[t]
            for k, M in enumerate(Ms[t]):
                prob = z[t][k]
                curMenu = {realJ[s] for s in M}
                choiceSet[i] |= curMenu
                randomMenuDict[i].append((curMenu, prob))

        return choiceSet, randomMenuDict, quota, numAllowed, lamda, gamma, xi


def solveCertain(
    name, budget1, max_frl, alpha, sizeBudget=0, card_budget=0, reOptimize=True
):
    # k=20

    cutoffFile = os.path.join(PlanFolders.PRIORITIES, "%s.csv" % name)
    plan = OptimizedPlan.sfusd2018K(
        averageBudget=budget1,
        max_frl=max_frl,
        alpha=alpha,
        sizeBudget=sizeBudget,
        cardBudget=card_budget,
        reOptimize=reOptimize,
    )
    # plan=OptimizedPlan.default2013K2Small(budget1, alpha,budget2,cardBudget=budget3,reOptimize=reOptimize)
    plan.saveCutoffs(cutoffFile)
    baseDir = os.path.join(PlanFolders.PRIORITIES, name)
    plan.saveRandomMenus(os.path.join(baseDir, "menus.csv"))
    plan.saveQuota(os.path.join(baseDir, "quota.csv"))
    plan.saveShadowPrice(os.path.join(baseDir, "shadowPrice.csv"))
    plan.outputAnalysis(os.path.join(baseDir, "analysis.csv"))
    paramFile = os.path.join(PlanFolders.PARAMS, "%sParams.csv" % name)
    ConvertPrioritiesParams().convert(paramFile, cutoffFile)


if __name__ == "__main__":
    dist_budget = 10
    max_frl = 1
    alpha = 0.5
    card_budget = 0
    name = f"dist{dist_budget}_maxfrl{max_frl}_alpha{alpha}_card{card_budget}_umodelavg"
    solveCertain(
        name=name,
        budget1=dist_budget,
        max_frl=max_frl,
        alpha=alpha,
        card_budget=card_budget,
    )
    print(f"Finished running {name}.")
