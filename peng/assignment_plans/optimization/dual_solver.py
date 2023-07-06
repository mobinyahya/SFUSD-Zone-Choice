import sys

from tqdm import tqdm

from peng.assignment_plans.optimization.constraints import *
from peng.assignment_plans.optimization.optimal_assortment import sociallyOptimalAssortment
from peng.assignment_plans.optimization.optimal_assortment_ext import sociallyOptimalAssortmentC
from peng.utils.common import _getArg
import gurobipy as guro
import time
from peng.constants.locations import Folders

import numpy as np
import os
from itertools import chain
from peng.utils import data_saver


class DualSolver(object):
    """Class that computes the dual of an LP of the form in Section 6.1 of the paper.
    """

    def __init__(self, logit, c, B, D, q, w, frl, max_frl, alpha=1, **kwargs):
        """Conventions:
        - c[t,j]=0 implies j is in the walk-zone of neighborhood t
        - k=0 implies no limit on cardinality outside of walk-zone
        - C=0 implies no limit on total expected cardinality of assortments outside of walkzone.
        """

        self._logit = logit
        self._caps = np.append(q, [0])   # capacities
        self._w = np.array(w, dtype=float)  # student counts
        self._totalw = float(sum(w))
        self._distance_budget = B  # miles bused budget
        self._cardinality_budget = D  # number of school options
        self._frl = frl
        self._max_frl = max_frl  # max frl percentage allowed

        k = _getArg(kwargs, "k", 0)
        self._allowed = _getArg(kwargs, "allowed", {})

        self._alpha = alpha

        # enforce capacity at all schools
        self._applyCap = np.ones((self._logit.T, self._logit.S), dtype=int)
        self._applyCap[:, -1] = 0

        c = self._applyCap * c
        self._dists = c  # miles bused (now distance)

        self._thresh = _getArg(kwargs, "thresh", 1e-5)
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

        self.m: guro.Model = None
        self.callback = None

    def _constructModel(self, Ms, verbose):
        m = guro.Model("LP dual")
        self.m = m

        m.ModelSense = 1
        if not verbose:
            m.setParam("OutputFlag", False)

        # Feasibility variables
        feasibilityVar = {}
        for t in tqdm(range(self.T), desc="feasibility", file=sys.stdout):
            feasibilityVar[t] = m.addVar(lb=-np.inf, obj=1.0, name="feasibility_%d" % t)

        # Distance variable
        if distActive:
            distVar = m.addVar(lb=0, obj=self._distance_budget, name="busing_constraint")
        else:
            distVar = m.addVar(lb=0, obj=0, name="busing_constraint")

        # Capacity variables
        capacityVar = {}
        print("TOTAL TYPES", sum(self.w))
        print("TOTAL CAPS", sum(self._caps))
        for s in tqdm(range(self.S), desc="capacities", file=sys.stdout):
            if capacityActive:
                capacityVar[s] = m.addVar(lb=0, obj=self._caps[s], name="capacity_%d" % s)
            else:
                capacityVar[s] = m.addVar(lb=0, obj=0, name="capacity_%d" % s)

        # Cardinality variables
        if cardActive and self._cardinality_budget > 0:
            cardVar = m.addVar(lb=0, obj=self._cardinality_budget, name="cardConstraint")
        else:
            cardVar = m.addVar(lb=0, obj=0, name="cardConstraint")

        # FRL variables
        frlVar = {}
        self.frl_rhs = self._max_frl * self._caps
        for s in tqdm(range(self.S), desc="FRL", file=sys.stdout):
            if frlActive:
                frlVar[s] = m.addVar(lb=0, obj=self.frl_rhs[s], name="frl_%d" % s)
            else:
                frlVar[s] = m.addVar(lb=0, obj=0, name="frl_%d" % s)

        variables = self.variables = Variables()
        variables.feasibilityVar = feasibilityVar

        variables.distVar = distVar
        variables.capacityVar = capacityVar
        variables.cardVar = cardVar
        variables.frlVar = frlVar

        self.x = {t: {} for t in range(self.T) if self._w[t] > 0}
        self.columns = {t: set() for t in self.x.keys()}
        for t in Ms:
            self._addConstraints(t, Ms[t])

        m.update()

    def _addCallback(self):
        self.m.addVar(vtype=guro.GRB.BINARY, name="dummy")
        self.m.setParam('LazyConstraints', 1)

        def callback_function(model, where):
            if where != guro.GRB.Callback.MIPSOL:
                return
            dist_price, cap_price, valid_probs_price, cardinality_price, frl_price = [
                self._parseValuesCB(variable) for variable in self.variables
            ]
            self._optAssortmentsCB(dist_price, cap_price, valid_probs_price, cardinality_price, frl_price)

        self.callback = callback_function

    def _getCardCapped(self, Ms):
        return {
            t: [(self._limit[t] * self._applyCap[t])[M].sum() for M in Ms[t]]
            for t in Ms
        }

    def _parseValues(self, variable):
        if type(variable) == guro.Var:
            return variable.X
        if type(variable) == list:
            return np.array([c.X for c in variable])
        elif type(variable) == dict:
            return np.array(
                [variable[i].X if variable[i] else 0.0 for i in range(len(variable))]
            )

    def _parseValuesCB(self, variable):
        m = self.m
        if type(variable) == guro.Var:
            return m.cbGetSolution(variable)
        if type(variable) == list:
            return np.array([m.cbGetSolution(c) for c in variable])
        elif type(variable) == dict:
            return np.array(
                [m.cbGetSolution(variable[i]) if variable[i] else 0.0 for i in range(len(variable))]
            )

    def _optAssortments(self, dist_price, cap_price, valid_probs_price, cardinality_price, frl_price):
        n = self.S
        new_cons = 0
        for t in tqdm(range(self.T), file=sys.stdout, desc="assortments"):
            if self._w[t] <= 0:
                continue

            v = self.v[t]

            # a (\alpha in paper) is the weight on the first term in eqn (31)
            # a = (self.alpha * self.w[t] + nu[t]) * self._logit.beta[t]
            a = self.alpha * self.w[t]  # seems like there should be a 1/\Lambda in first term
            r = - self.w[t] * (self._dists[t] * dist_price + self._applyCap[t] * cap_price + self._frl[t] * frl_price)  # seems like there should be a 1/n in the first term
            cost = self._cost * self.w[t] * cardinality_price  # remove area cost, last term of objective in eqn (31), seems like there should be a 1/\Lambda

            allowed = self.allowed(t)
            mandatory = self._mandatory[t]
            limit = self._limit[t]

            # mu_t, optM, opts = sociallyOptimalAssortment(n, v, r, cost, allowed, mandatory, limit, a, verbose=False)  # python version
            mu_t, optM = sociallyOptimalAssortmentC(n, v, r, cost, allowed, mandatory, limit, a, verbose=False)  # C version
            opts = [optM]

            if mu_t > valid_probs_price[t] + self._thresh:
                repeated = tuple(optM) in self.columns[t]
                new_cons += self._addConstraints(t, opts)

                if repeated and self.callback is None:
                    M = optM
                    v2, p = self.logit.vp(t, M)
                    card = self._limit[t][M].sum()

                    print("error: repeated constraint")
                    print("unassigned school:", self.S-1)
                    print("t:", t)
                    print("S:", M)
                    print("p:", p)
                    print("mu_t:", mu_t, ">", valid_probs_price[t])
                    print("alpha_t*U_t(S):", a*v2)

                    print(["feas_%s" % t], "=", [valid_probs_price[t]], [1.0])
                    print(["cap_%s" % s for s in M], "=", [cap_price[s] for s in M], p*self.w[t]*self._applyCap[t, M])
                    print(["dist"], "=", [dist_price], [np.dot(p, self._dists[t, M]) * self.w[t]])
                    print(["frl_%s" % s for s in M], "=", [frl_price[s] for s in M], p*self.w[t]*self._frl[t])
                    print(["card"], "=", [cardinality_price], [self.w[t]*card])

                    mu2, M2, _ = sociallyOptimalAssortment(n, v, r, cost, allowed, mandatory, limit, a, verbose=True)
                    print(mu2, M2)

                    exit()

        self.m.update()
        return new_cons

    def _optAssortmentsCB(self, dist_price, cap_price, valid_probs_price, cardinality_price, frl_price):
        variables = self.variables
        n = self.S
        new_cons = 0
        for t in tqdm(range(self.T), file=sys.stdout, desc="assortments CB"):
            if self._w[t] <= 0:
                continue

            v = self.v[t]
            a = self.alpha * self.w[t]  # seems like there should be a 1/\Lambda in first term
            r = - self.w[t] * (self._dists[t] * dist_price + self._applyCap[t] * cap_price + self._frl[t] * frl_price)  # seems like there should be a 1/n in the first term
            cost = self._cost * self.w[t] * cardinality_price  # remove area cost, last term of objective in eqn (31), seems like there should be a 1/\Lambda

            allowed = self.allowed(t)
            mandatory = self._mandatory[t]
            limit = self._limit[t]

            mu_t, M = sociallyOptimalAssortmentC(n, v, r, cost, allowed, mandatory, limit, a, verbose=False)  # C version

            if mu_t <= valid_probs_price[t] + self._thresh:
                continue
            if tuple(M) in self.columns[t]:
                continue

            v, p = self.logit.vp(t, M)
            expr = guro.LinExpr()
            if feasibilityActive:
                expr.addTerms([1.0], [variables.feasibilityVar[t]])
            if capacityActive:
                expr.addTerms(p * self.w[t] * self._applyCap[t, M], [variables.capacityVar[s] for s in M])
            if distActive:
                expr.addTerms([np.dot(p, self._dists[t, M]) * self.w[t]], [variables.distVar])
            if cardActive and self._cardinality_budget > 0:
                card = self._limit[t][M].sum()
                expr.addTerms([self.w[t] * card], [variables.cardVar])
            if frlActive:
                expr.addTerms(p * self.w[t] * self._frl[t], [variables.frlVar[s] for s in M])

            xt = self.x[t]
            k = len(xt)
            xt[k] = self.m.cbLazy(expr >= self.alpha * v * self.w[t])
            self.columns[t].add(tuple(M))
            new_cons += 1
        print("new cons:", new_cons)

    def _updateMenus(self, t, Ms):
        for M in Ms:
            cur = np.zeros(self.S, dtype=bool)
            cur[M] = 1
            self._inMenu[t].append(cur)

    def _addConstraints(self, t, Ms):
        variables = self.variables
        m = self.m
        new_cons = 0
        xt = self.x[t]
        k = len(xt)
        Ms_new = []
        for M in Ms:
            if tuple(M) in self.columns[t]:
                continue
            Ms_new.append(M)

            v, p = self.logit.vp(t, M)

            expr = guro.LinExpr()

            # add term to valid probabilities constraint
            if feasibilityActive:
                expr.addTerms([1.0], [variables.feasibilityVar[t]])

            # add term to capacity constraint for any school on the menu
            if capacityActive:
                # guro.quicksum(x[t][j] * p[t, j, s] * self.w[t] * self._applyCap[t, s] for t, j in indices)
                expr.addTerms(p * self.w[t] * self._applyCap[t, M], [variables.capacityVar[s] for s in M])

            # add term to distance constraint
            if distActive:
                expr.addTerms([np.dot(p, self._dists[t, M]) * self.w[t]], [variables.distVar])

            # add term to cardinality constraint
            if cardActive and self._cardinality_budget > 0:
                card = self._limit[t][M].sum()
                expr.addTerms([self.w[t] * card], [variables.cardVar])

            # add terms to FRL constraints for relevant schools
            if frlActive:
                expr.addTerms(p * self.w[t] * self._frl[t] * self._applyCap[t, M], [variables.frlVar[s] for s in M])

            xt[k] = m.addConstr(expr >= self.alpha * v * self.w[t])
            self.columns[t].add(tuple(M))
            new_cons += 1
            k += 1

        self._updateMenus(t, Ms_new)
        return new_cons

    def allowed(self, t):
        if t in self._allowed:
            return self._allowed[t]
        else:
            return range(self.S)

    def _genInitial(self):
        Ms = {t: [[self.S - 1]] for t in range(self.T) if self._w[t] > 0}

        return Ms

    def _printObj(self, primalObj):
        print("Objectives")
        print("\tPrimal Objective %.7f" % primalObj)

    def _extractPolicy(self):
        z = {}
        Ms = {}
        for t in self.x:
            zt = []
            Mst = []
            xt = self.x[t]
            menuIndicator = self._inMenu[t]
            for j in xt:
                val = xt[j].Pi
                if val > 0:
                    zt.append(val)
                    Mst.append(list(np.where(menuIndicator[j])[0]))
            z[t] = zt
            Ms[t] = Mst

        return z, Ms

    def solve(self, verbose=True, write=False):
        Ms = self._genInitial()

        self._constructModel(Ms, verbose)
        # self._addCallback()
        m = self.m
        x = self.x
        self.v = {t: self.logit.w(t) for t in x}

        # m.write(os.path.join(Folders.LOGS,'Initial.lp'))
        startTime = lastTime = time.perf_counter()

        while True:
            if verbose:
                print("**********Solving LP")
            if self.callback is None:
                m.optimize()
            else:
                m.optimize(self.callback)
            if m.status != guro.GRB.status.OPTIMAL:
                m.write(os.path.join(Folders.LOGS, "Infeasible.lp"))
                raise ValueError("Gurobi not optimal, status %s" % m.status)
            if write:
                m.write(os.path.join(Folders.LOGS, "mostRecent.lp"))
                m.write(os.path.join(Folders.LOGS, "mostRecent.sol"))
            primalObj = m.ObjVal

            dist_price, cap_price, valid_probs_price, cardinality_price, frl_price = [
                self._parseValues(variable) for variable in self.variables
            ]

            new_cons = self._optAssortments(dist_price, cap_price, valid_probs_price, cardinality_price, frl_price)

            if verbose:
                self._printObj(primalObj)
                print("Time elapsed ")
                print("\t\t since last cycle %.3f" % (time.perf_counter() - lastTime))
                print("\t\t since beginning %.3f" % (time.perf_counter() - startTime))
                lastTime = time.perf_counter()

            print("new constraints:", new_cons)
            if new_cons == 0:
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
            base = self.alpha
            totalCost = dist_price * self._dists[t] + cap_price * self._applyCap[t]
            myList = []
            for j in range(len(Ms[t])):
                myList.append(
                    base
                    + np.dot(totalCost[Ms[t][j]], self.logit.vp(t, Ms[t][j], pOnly=True))
                )
            quota[t] = myList
        numAllowed = self._getCardCapped(Ms)

        return z, Ms, quota, numAllowed, valid_probs_price, None, cap_price, dist_price, cardinality_price, frl_price

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
                z[t][j] * np.dot(p[t, j], self._dists[t]) * self.w[t]
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
    def logit(self):
        return self._logit

    @property
    def w(self):
        return self._w