"""
Created on Jul 19, 2019

@author: pengshi
"""
import numpy as np

eps = 1e-6
factor = 1e-4


def sociallyOptimalAssortment(
    n, v, r, cost, allowed, mandatory, limit, a, verbose=False
):
    """This function implements a modification of the algorithm for socially optimal assortment planning 
    for MNL utilities and cardinality constraints (Algorithm 3). Instead of having a cardinality constraint,
    this function penalizes the objective by an array called cost, which specifies for each cardinality
    how much to penalize the objective. For example, if cost[j]=0 for all j<=k, and cost[j]=infinity for 
    j>k, then it implements the cardinality constraint k.

    Input Data:
        - n: the number of items.
        - v: an array of the attraction weights.
        - r: an array of the revenue of each item.
        - cost: an array indicating how much to penalize for each cardinality of the assortment. Only
                items j with limit[j]=1 are counted in the cardinality.
        - mandatory: a Boolean array whether each item must be included in the assortment or not.
        - limit: a Boolean array for whether each item counts toward the cardinality used in the cost.
        - a,b: scalar parameters indicating the weights of the expected utility term and the revenue term.
        - verbose=False: whether to print debugging outputs.
    
    Returned Values:         
        - optZ: the optimal objective value.
        - optM: a list specifying the index of the items in an optimal assortment.    
    """

    v = np.array(v)
    r = np.array(r)
    limit = np.array(limit)
    vr = v * r
    optionalList = [i for i in allowed if not mandatory[i]]
    limitList = [i for i in optionalList if limit[i]]

    tmp = sorted([(v[i], r[i], -i) for i in limitList], reverse=True)
    jOrder = [-entry[-1] for entry in tmp]
    o = np.zeros(n, dtype=int)
    kmax = int(limit[list(allowed)].sum())
    kmin = int(limit[mandatory].sum())
    vsum = np.zeros(kmax + 1)
    vrsum = np.zeros(kmax + 1)

    initialM = np.array(mandatory)
    for i in allowed:
        if not limit[i]:
            initialM[i] = 1

    indices = np.arange(n)

    M = {}
    opts = []
    vsum[kmin] = v[initialM].sum()
    vrsum[kmin] = vr[initialM].sum()
    M[kmin] = np.array(initialM)
    optZ = -np.inf
    optM = initialM

    if verbose:
        print("initial:")
        print(f"n={n}, kmin={kmin}, kmax={kmax}, M[kmin]={indices[M[kmin]]}, optM={indices[initialM]}")
        print(f"vsum[kmin]={vsum[kmin]}, vsum[kmax]={vsum[kmax]}")
        print()

    def update(optZ, optM, k, vsum, vrsum, M, where=""):
        z = a * np.log(vsum[k]) + vrsum[k] / vsum[k] - cost[k]
        # if verbose:
        #     print(f"COMPUTED z={z} k={k} M={list(indices[M[k]])} from {where}")
        if (z > optZ + eps) and (len(optM) > 0):
            if verbose:
                print(f"\tUPDATING using k={k} vsum[k]={vsum[k]} vrsum[k]={vrsum[k]} optZ={optZ} -> {z}")
                print(f"\toptM={list(indices[optM])} -> {list(indices[M[k]])}")
                print(f"\tv     : {list(v[M[k]])}")
                print(f"\tlog(v): {list(np.log(v[M[k]]))}")
                print()

            opts.append((z, M[k]))
            return z, np.array(M[k])
        return optZ, optM

    # optZ, optM = update(optZ, optM, kmin, vsum, vrsum, M)
    for i, j in enumerate(jOrder):
        o[j] = i + 1
        k = kmin + i + 1
        vsum[k] = vsum[k - 1] + v[j]
        vrsum[k] = vrsum[k - 1] + vr[j]
        M[k] = np.array(M[k - 1])
        M[k][j] = 1
        optZ, optM = update(optZ, optM, k, vsum, vrsum, M, where="jOrder")

    # step (1) from Algorithm 3
    tau = sorted(
        [
            ((vr[i] - vr[j]) / (v[i] - v[j]), -i, j)
            for i in limitList
            for j in limitList
            if v[i] > v[j]
        ]
        + [(r[i], -i, n) for i in optionalList]
    )

    # steps (3) and (4) from Algorithm 3, run together
    for unused, minusI, j in tau:
        i = -minusI
        # if verbose:
        #     print('\no=', o, '\nvsum=', vsum, '\nvrsum=', vrsum, '\nM=', {k:indices[M[k]] for k in M}, '\n')
        #     print(f'lamda={unused} i={i} j={j}')

        if j == n:
            for k in range(kmin, kmax + 1):
                if M[k][i]:
                    # if verbose:
                    #     print(f"Taking out i={i} with o[i]={o[i]}")
                    M[k][i] = 0
                    # vsum[k] -= v[i]
                    # vrsum[k] -= vr[i]
                    vsum[k] = v[M[k]].sum()
                    vrsum[k] = vr[M[k]].sum()
                    optZ, optM = update(optZ, optM, k, vsum, vrsum, M, where="j == n")

        elif o[i] < o[j]:
            o[i], o[j] = o[j], o[i]
            k = o[j] + kmin
            if M[k][i]:
                # if verbose:
                #     print(f"Swapping out i={i} with o[i]={o[i]} with j={j} and o[j]={o[j]}")
                M[k][j] = 1
                M[k][i] = 0
                # vsum[k] += v[j] - v[i]
                # vrsum[k] += vr[j] - vr[i]
                vsum[k] = v[M[k]].sum()
                vrsum[k] = vr[M[k]].sum()
                optZ, optM = update(optZ, optM, k, vsum, vrsum, M, where="oi < oj")

    if verbose:
        print(
            "\no=", o,
            "\nvsum=", vsum,
            "\nvrsum=", vrsum,
            "\nM=", {k: list(indices[M[k]]) for k in M},
            "\n",
        )

    # return optZ, list(indices[optM]), [list(indices[o]) for v, o in opts if v >= cutoff]
    return optZ, list(indices[optM])
