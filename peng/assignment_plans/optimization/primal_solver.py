import os
import sys
import time
from itertools import chain

import gurobipy as guro
import numpy as np
from tqdm import tqdm

from peng.assignment_plans.optimization.constraints import *
from peng.assignment_plans.optimization.optimal_assortment_ext import sociallyOptimalAssortmentC
from peng.constants.locations import Folders
from peng.utils import data_saver
from peng.utils.common import _getArg


class PrimalSolver(object):
    """Class that computes an LP of the form in Section 6.1 of the paper.
    """

    def __init__(self, logit, c, B, D, q, w, frl, max_frl, alpha=1, sibling_frl=None, original_capacity=None, **kwargs):
        """Conventions:
        - c[t,j]=0 implies j is in the walk-zone of neighborhood t
        - k=0 implies no limit on cardinality outside of walk-zone
        - C=0 implies no limit on total expected cardinality of assortments outside of walkzone.
        """

        self._logit = logit
        self._caps = np.append(q, [0])  # capacities  TODO: fix
        self._w = np.array(w, dtype=float)  # student counts
        self._totalw = float(sum(w))
        self._distance_budget = B  # miles bused budget
        self._cardinality_budget = D  # number of school options
        self._max_frl = max_frl  # max frl percentage allowed
        self._frl = frl
        self._sibling_frl = np.append(sibling_frl, [0])
        self._original_caps = np.append(original_capacity, [0])

        k = _getArg(kwargs, "k", 0)
        self._allowed = _getArg(kwargs, "allowed", {})

        self._alpha = alpha

        # enforce capacity at all schools
        self._applyCap = np.ones((self._logit.T, self._logit.S), dtype=int)
        self._applyCap[:, -1] = 0

        c = self._applyCap * c
        self._dists = c  # miles bused (now distance)

        self._thresh = _getArg(kwargs, "thresh", 1e-5)
        self.columns = {t: set() for t in range(self.T)}
        self._inMenu = {t: [] for t in range(self.T)}

        self._mandatory = {
            t: np.array(1 - self._applyCap[t], dtype=bool)
            for t in range(self.T)
        }
        sch_limit = np.ones(self.S)
        sch_limit[-1] = 0
        self._limit = {t: sch_limit for t in range(self.T)}  # school counts towards cardinality
        self._cost = np.arange(self.T + 1)
        if k > 0:
            self._cost[(k + 1):] = np.inf

        if not os.path.exists(Folders.LOGS):
            os.makedirs(Folders.LOGS)

    def _dualObj(self, valid_probs_price, dist_price, cap_price, cardinality_price, frl_price):
        return (
                np.sum(valid_probs_price)
                + np.dot(cap_price, self._caps)
                + dist_price * self._distance_budget
                + cardinality_price * self._cardinality_budget
                + np.dot(frl_price, self.frl_rhs)
        )

    def _constructModel(self, Ms, verbose):
        x = {t: {} for t in range(self.T) if self._w[t] > 0}
        m = guro.Model("LP")
        m.ModelSense = -1
        if not verbose:
            m.setParam("OutputFlag", False)

        # Adding variables
        y = m.addVar(lb=-np.inf, obj=(1 - self.alpha) * self._totalw, name="y")
        zero = guro.quicksum([])

        # Feasibility constraints
        feasibilityConstr = {}
        for t in tqdm(range(self.T), desc="feasibility", file=sys.stdout):
            feasibilityConstr[t] = m.addConstr(zero == 1, name="feasibility_%d" % t)

        if distActive:
            distConstr = m.addConstr(zero <= self._distance_budget, name="busing_constraint")
        else:
            distConstr = m.addConstr(0 <= 1, name="busing_constraint")

        # Capacity constraints
        capacityConstr = []
        print("TOTAL TYPES", sum(self.w))
        print("TOTAL CAPS", sum(self._caps))
        for s in tqdm(range(self.S), desc="capacities", file=sys.stdout):
            if capacityActive:
                capacityConstr.append(m.addConstr(zero <= self._caps[s], name="capacity_%d" % s))
            else:
                capacityConstr.append(m.addConstr(0 <= 1, name="capacity_%d" % s))

        # Equity constraints
        equityConstr = {}
        for t in tqdm(range(self.T), desc="equity", file=sys.stdout):
            if t in x:
                equityConstr[t] = m.addConstr(y <= zero, name="equity_%d" % t)
            else:
                equityConstr[t] = 0

        # Cardinality constraint
        sizeConstr = m.addConstr(0 <= 1, name="sizeConstraint")
        if cardActive and self._cardinality_budget > 0:
            cardConstr = m.addConstr(zero <= self._cardinality_budget, name="cardConstraint")
        else:
            cardConstr = m.addConstr(0 <= 1, name="cardConstraint")

        # FRL constraints
        frl_constr = []

        if self._sibling_frl is not None:
            self.frl_rhs = self._max_frl * (self._original_caps - self._caps) - self._sibling_frl
            # if self.frl_rhs.min() < 0:
            #     raise ValueError("FRL constraint violated by preassigned siblings.")
        else:
            self.frl_rhs = np.zeros(len(self._caps))  # self._max_frl * self._caps
        print("FRL LIMITS", self.frl_rhs)
        for s in tqdm(range(self.S), desc="FRL", file=sys.stdout):
            if frlActive:
                frl_constr.append(m.addConstr(zero <= self.frl_rhs[s], name="frl_%d" % s))
            else:
                frl_constr.append(m.addConstr(0 <= 1, name="frl_%d" % s))

        m.update()

        constraints = Constraints()
        constraints.feasibilityConstr = feasibilityConstr
        constraints.equityConstr = equityConstr

        constraints.distConstr = distConstr
        constraints.sizeConstr = sizeConstr
        constraints.capacityConstr = capacityConstr
        constraints.cardConstr = cardConstr
        constraints.frl_constr = frl_constr

        self.x = x
        self.y = y
        self.m = m
        self.constraints = constraints

        res = 0
        for t in Ms:
            res += self._addColumns(t, Ms[t])
        print("new cols:", res)

        m.update()

    def _getCard(self, Ms):
        return {t: [self._limit[t][M].sum() for M in Ms[t]] for t in Ms}

    def _getCardCapped(self, Ms):
        return {
            t: [(self._limit[t] * self._applyCap[t])[M].sum() for M in Ms[t]]
            for t in Ms
        }

    def _parseShadowPrices(self, constr):
        if type(constr) == guro.Constr:
            return constr.Pi
        if type(constr) == list:
            return np.array([c.Pi for c in constr])
        elif type(constr) == dict:
            return np.array(
                [constr[i].Pi if constr[i] else 0.0 for i in range(len(constr))]
            )

    def _optAssortments(self, dist_price, cap_price, valid_probs_price, min_util_price, cardinality_price, frl_price):
        mu2 = np.zeros(self.T)
        Ms2 = {}
        n = self.S
        new_cols = 0
        for t in tqdm(range(self.T), file=sys.stdout, desc="optimal assortments"):
            if self._w[t] <= 0:
                continue

            v = self.logit.w(t)

            # a (\alpha in paper) is the weight on the first term in eqn (31)
            a = (self.alpha * self.w[t] + min_util_price[t]) * self._logit.beta[
                t]  # seems like there should be a 1/\Lambda in first term

            r = - self.w[t] * (dist_price * self.c[t] + cap_price * self._applyCap[t] + frl_price * self._frl[
                t])  # seems like there should be a 1/n in the first term

            # cost = self._cost * (xi[0] * self._s[t] + xi[1] * self.w[t]
            cost = self._cost * (cardinality_price * self.w[
                t])  # remove area cost, last term of objective in eqn (31), seems like there should be a 1/\Lambda

            allowed = self.allowed(t)
            mandatory = self._mandatory[t]
            limit = self._limit[t]

            # mu2[t], optM = sociallyOptimalAssortment(n, v, r, cost, allowed, mandatory, limit, a, verbose=False)  # python version
            mu2[t], optM = sociallyOptimalAssortmentC(n, v, r, cost, allowed, mandatory, limit, a,
                                                      verbose=False)  # C version
            Ms2[t] = [optM]

            if mu2[t] < valid_probs_price[t] - self._thresh:
                # mu2[t],Ms=socially_optimal_assortment(n,v,r,cost,allowed,mandatory,limit,a,b,verbose=True)
                totalCost = -r
                origZ, origMs = self._extractPolicy()
                print("Error dump:")
                print("t", t)
                print("orignal value", valid_probs_price[t])
                print("new value", mu2[t])

                print("a= %s \t gamma=%s \t xi=%s" % (a, dist_price, cardinality_price))

                print("allowed", self.allowed(t))
                print("orignal probs", origZ[t])
                print(
                    "original menus",
                    origMs[t],
                    [
                        a * self.logit.vp(t, M, vOnly=True)
                        - np.dot(totalCost[M], self.logit.vp(t, M, pOnly=True))
                        for M in origMs[t]
                    ],
                )
                print(
                    "new menu",
                    optM,
                    a * self.logit.vp(t, optM, vOnly=True)
                    - np.dot(totalCost[optM], self.logit.vp(t, optM, pOnly=True)),
                )
                raise ValueError("Subproblem did not meet original!")

        print("add new columns")
        for t in range(self.T):
            if mu2[t] > valid_probs_price[t] + self._thresh:
                new_cols += self._addColumns(t, Ms2[t])

        self.m.update()
        return mu2, Ms2, new_cols

    def _updateMenus(self, t, Ms):
        for M in Ms:
            cur = np.zeros(self.S, dtype=bool)
            cur[M] = 1
            self._inMenu[t].append(cur)

    def _addColumns(self, t, Ms):
        cstr = self.constraints

        new_cols = 0
        xt = self.x[t]
        k = len(xt)
        Ms_new = []
        for M in Ms:
            if tuple(M) in self.columns[t]:
                continue
            Ms_new.append(M)

            v, p = self.logit.vp(t, M)
            col = guro.Column()

            # add term to valid probabilities constraint
            if feasibilityActive:
                col.addTerms(1, cstr.feasibilityConstr[t])

            # add term to distance constraint
            if distActive:
                col.addTerms(np.dot(p, self.c[t, M]) * self.w[t], cstr.distConstr)

            # add term to capacity constraint for any school on the menu
            if capacityActive:
                col.addTerms(p * self.w[t] * self._applyCap[t, M], [cstr.capacityConstr[s] for s in M])

            # add term to minimum neighborhood utility constraint (constraint with >=, hence negative sign)
            col.addTerms(-v, cstr.equityConstr[t])

            card = self._limit[t][M].sum()

            # add term to cardinality/menu size constraint
            if cardActive:
                if self._cardinality_budget > 0:
                    col.addTerms(self.w[t] * card, cstr.cardConstr)

            # add terms to FRL constraints for relevant schools
            if frlActive:
                col.addTerms(p * self.w[t] * self._frl[t] * self._applyCap[t, M] - self._max_frl * (
                            p * self.w[t] * self._applyCap[t, M]), [cstr.frl_constr[s] for s in M])
            xt[k] = self.m.addVar(lb=0, obj=self.alpha * v * self.w[t], column=col)
            k += 1
            new_cols += 1

        self._updateMenus(t, Ms_new)
        return new_cols

    def allowed(self, t):
        if t in self._allowed:
            return self._allowed[t]
        else:
            return range(self.S)

    def _genInitial(self):
        Ms = {t: [[self.S - 1]] for t in range(self.T) if self._w[t] > 0}
        return Ms

    def _printObj(self, primalObj, dualObj):
        print("Objectives")
        print("\tPrimal Objective %.7f" % primalObj)
        print("\tDual Objective %.7f" % dualObj)
        print("\tGap %.7f" % (dualObj - primalObj))

    def _printDual(self, gamma, lamda, mu, nu, xi):
        print("Dual variables")
        print("\t busing (gamma):%s" % gamma)
        print("\t capacity (lamda):%s" % lamda)
        print("\t feasibility (mu):%s" % mu)
        print("\t equity (nu):%s" % nu)
        print("\t size (xi):%s" % xi)

    def _extractPolicy(self):
        z = {}
        Ms = {}
        for t in self.x:
            zt = []
            Mst = []
            xt = self.x[t]
            menuIndicator = self._inMenu[t]
            for j in xt:
                if xt[j].X > 0:
                    zt.append(xt[j].X)
                    Mst.append(list(np.where(menuIndicator[j])[0]))
            z[t] = zt
            Ms[t] = Mst

        return z, Ms

    def solve(self, verbose=True, write=False):
        Ms = self._genInitial()

        self._constructModel(Ms, verbose)
        # m.write(os.path.join(Folders.LOGS,'Initial.lp'))
        dualObj = np.Inf
        startTime = lastTime = time.time()

        m = self.m
        # m.Params.DualReductions = 0
        # m.computeIIS()
        # m.write("model_iis.ilp")
        while True:
            if verbose:
                print("**********Solving LP")
            m.optimize()
            if m.status != guro.GRB.status.OPTIMAL:
                m.write(os.path.join(Folders.LOGS, "Infeasible.lp"))
                raise ValueError("Gurobi not optimal, status %s" % m.status)
            if write:
                m.write(os.path.join(Folders.LOGS, "mostRecent.lp"))
                m.write(os.path.join(Folders.LOGS, "mostRecent.sol"))
            primalObj = m.ObjVal

            dist_price, cap_price, valid_probs_price, min_util_price, size_price, cardinality_price, frl_price = [
                self._parseShadowPrices(constr) for constr in self.constraints
            ]
            mu2, Ms2, new_cols = self._optAssortments(dist_price, cap_price, valid_probs_price, min_util_price,
                                                      cardinality_price, frl_price)

            print("Old dual objective", dualObj)
            print("New dual objective", self._dualObj(mu2, dist_price, cap_price, cardinality_price, frl_price))
            dualObj = min(dualObj, self._dualObj(mu2, dist_price, cap_price, cardinality_price, frl_price))

            if verbose:
                self._printObj(primalObj, dualObj)
                print("Time elapsed ")
                print("\t\t since last cycle %.3f" % (time.time() - lastTime))
                print("\t\t since beginning %.3f" % (time.time() - startTime))
                lastTime = time.time()

            if not primalObj < dualObj + self._thresh:
                print("Primal > Dual! Something is wrong")
                print("Weights ", self._w)
                self._printDual(dist_price, cap_price, valid_probs_price, min_util_price, cardinality_price)
                print("\t new feasibility (valid_probs_price):%s" % mu2)

                print("\t old Menus", Ms)
                print("\t new Menus", Ms2)

                print("Primal Objective %s:" % primalObj)
                print("old dual Objective %s:" % self._dualObj(valid_probs_price, dist_price, cap_price, size_price,
                                                               cardinality_price, frl_price))
                print(
                    "cur dual Objective %s:" % self._dualObj(mu2, dist_price, cap_price, cardinality_price, frl_price))
                print("best so far dual Objective %s:" % dualObj)
                raise ValueError("Primal > Dual! Something is wrong!")

            print("new columns:", new_cols)
            if new_cols == 0:
                break

        z, Ms = self._extractPolicy()

        if verbose:
            self.analyzeMenus(z, Ms)
            self.analyzeSize(Ms)

        # The quota is the negative of the optimal \lambda in \max v_i(r_i-\lambda)
        # Used to define the optimal assortment in an intuitive way.
        quota = {}
        for t in z:
            # base=self.logit.beta[t]*(self.alpha+min_util_price[t]/self.w[t])
            base = self.alpha + min_util_price[t] / self.w[t]
            totalCost = dist_price * self._dists[t] + cap_price * self._applyCap[t]
            myList = []
            for j in range(len(Ms[t])):
                myList.append(
                    base
                    + np.dot(totalCost[Ms[t][j]], self.logit.vp(t, Ms[t][j], pOnly=True))
                )
            quota[t] = myList
        numAllowed = self._getCardCapped(Ms)

        return z, Ms, quota, numAllowed, valid_probs_price, min_util_price, cap_price, dist_price, cardinality_price, frl_price

    def analyzeSize(self, Ms):
        card = 0
        for t in Ms:
            menu = set(chain.from_iterable(Ms[t]))
            k = sum(self._limit[t][j] for j in menu)
            card += self.w[t] * k

        print("\tAverage busing choices", card / sum(self.w))

    def objective(self, z, Ms, normalized=False):
        ans = 0
        y = np.Inf
        totalw = float(np.sum(self.w))
        for t in range(self.T):
            vt = 0
            for j in range(len(Ms[t])):
                vt += self.logit.vp(t, Ms[t][j], vOnly=True) * z[t][j]
            ans += self.alpha * vt * self.w[t]
            y = min(y, vt)
        ans += (1 - self.alpha) * y * totalw
        if normalized:
            ans /= totalw
        return ans

    def _generateP(self, Ms):
        T, J, S = self.T, self.S, self.S
        p = np.zeros((T, J, S), dtype=float, order="F")
        for t in Ms:
            for j in range(len(Ms[t])):
                p[t, j][Ms[t][j]] = self.logit.vp(t, Ms[t][j], pOnly=True)
        return p

    def _generateV(self, Ms):
        v = {}
        for t in Ms:
            for j in range(len(Ms[t])):
                v[t, j] = self.logit.vp(t, Ms[t][j], vOnly=True)
        return v

    def extractCutoffs(self, z, Ms):
        cutoff = np.zeros((self.T, self.S))
        for t in Ms:
            for j in range(len(Ms[t])):
                cutoff[t, Ms[t][j]] += z[t][j]
        return cutoff

    def outputMenus(self, outFile, z, Ms, iind=None, jind=None, outputP=False):
        if iind is None:
            iind = {i: self.logit.iind(i) for i in self.logit.iset}
        if jind is None:
            jind = {j: self.logit.jind(j) for j in self.logit.jset}

        jList = sorted([(jind[j], j) for j in jind])

        header = ["Type", "Prob", "Value"] + [entry[1] for entry in jList]
        S = len(jList)
        lines = []
        v = self._generateV(Ms)
        if outputP:
            p = self._generateP(Ms)
        for i in iind:
            t = iind[i]
            for l in range(len(z[t])):
                prob = z[t][l]
                M = Ms[t][l]

                if outputP:
                    vec = p[t][l]
                else:
                    vec = np.zeros(S)
                    vec[M] = 1

                line = [i, prob, v[t, l]] + list(vec)
                lines.append(line)
        lines = sorted(lines)
        data_saver.saveLines(outFile, lines, header)

    def analyzeMenus(self, z, Ms):
        v = self._generateV(Ms)
        p = self._generateP(Ms)

        averageUtility = 0
        minUtility = np.Inf

        euler = 0.577215664901532

        for t in z:
            ut = 0
            beta = self.logit.beta[t]
            for j in range(len(z[t])):
                ut += z[t][j] * (v[t, j] + euler * beta)
            averageUtility += ut * self.w[t]
            minUtility = min(minUtility, ut)
        totalw = float(sum(self.w))
        averageUtility /= totalw
        averageBusing = (
                np.sum(
                    z[t][j] * np.dot(p[t, j], self.c[t]) * self.w[t]
                    for t in Ms
                    for j in range(len(Ms[t]))
                )
                / totalw
        )

        print("\tAverage utility: %s" % averageUtility)
        print("\tMinimum utility: %s" % minUtility)
        print("\tAverage Busing: %s" % averageBusing)
        print(
            "\tNormalized Objective: %s"
            % (averageUtility * self.alpha + (1 - self.alpha) * minUtility)
        )
        return averageUtility, minUtility, averageBusing

    @property
    def T(self):
        return self._logit.T

    @property
    def S(self):
        return self._logit.S

    @property
    def alpha(self):
        return self._alpha

    @property
    def c(self):
        return self._dists

    @property
    def logit(self):
        return self._logit

    @property
    def w(self):
        return self._w
