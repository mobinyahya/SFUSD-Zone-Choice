from ctypes import c_int, c_double
import numpy as np
import pathlib

optZ_type = np.ctypeslib.ndpointer(float, ndim=1, flags='aligned, contiguous, writeable')
optM_type = np.ctypeslib.ndpointer(bool, ndim=1, flags='aligned, contiguous, writeable')

double_const_type = np.ctypeslib.ndpointer(float, ndim=1, flags='aligned, contiguous')
int_const_type = np.ctypeslib.ndpointer(int, ndim=1, flags='aligned, contiguous')

path = pathlib.Path().absolute().parent.parent.parent / "external"
lib = np.ctypeslib.load_library('external.so', path)

lib.iter.restype = None
lib.iter.argtypes = [
    optZ_type, optM_type,  # double* optZ, bool* optM,
    c_int,  # const int n,
    double_const_type, double_const_type,  # const double* v, const double* vr, const double *r,
    c_double, double_const_type,  # const double a, const double* cost,
    c_int, c_int,  # const int kmin, const int kmax
    int_const_type, int_const_type,  # const pyint* limitlist, const pyint* optionallist
    c_int, c_int  # int n_limitlist, int n_optionallist
]

flags = ['ALIGNED', 'C_CONTIGUOUS']


def sociallyOptimalAssortmentC(
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

    cost = np.require(cost, float, flags)
    v = np.array(v, dtype=float, order="C")
    r = np.array(r, dtype=float, order="C")

    allowed = np.array(list(allowed), dtype=int)
    limit = np.array(limit, dtype=bool)

    optionalList = np.array([i for i in allowed if not mandatory[i]], dtype=int, order="C")
    limitList = np.array([i for i in optionalList if limit[i]], dtype=int, order="C")

    kmax = limit[allowed].sum()
    kmin = limit[mandatory].sum()

    optZ = np.empty(1, dtype=float, order="C")
    optM = np.array(mandatory, dtype=bool, order="C")
    optM[allowed] += ~limit[allowed]

    lib.iter(
        optZ, optM,  # double* optZ, bool* optM,
        n,  # const int n,
        v, r,  # const double* v, const double* vr, const double *r,
        a, cost,  # const double a, const double b, const double* cost,
        kmin, kmax,  # const int n, const int kmin, const int kmax
        limitList, optionalList,  # const pyint* limitlist, const pyint* optionallist
        len(limitList), len(optionalList)  # int n_limitlist, int n_optionallist
    )

    indices = np.arange(n)
    return optZ[0], list(indices[optM])
