from ctypes import c_int
import numpy as np

optZ_type = np.ctypeslib.ndpointer(float, ndim=1, flags='aligned, contiguous, writeable')
optM_type = np.ctypeslib.ndpointer(bool, ndim=2, flags='aligned, contiguous, writeable')

int_const = np.ctypeslib.ndpointer(int, ndim=1, flags='aligned, contiguous')
double_const = np.ctypeslib.ndpointer(float, ndim=1, flags='aligned, contiguous')
double_const_2d = np.ctypeslib.ndpointer(float, ndim=2, flags='aligned, contiguous')
bool_const_2d = np.ctypeslib.ndpointer(bool, ndim=2, flags='aligned, contiguous')

lib = np.ctypeslib.load_library('external.so', 'external/')

lib.iter_vec.restype = None
lib.iter_vec.argtypes = [
    optZ_type, optM_type,  # double* optZ, pybool* optM,
    c_int, c_int, c_int,  # const int T, const int n, const int U,
    double_const_2d, double_const_2d,  # const double* v, const double *r,
    double_const, double_const, double_const_2d,  # const double* a, const double* b, const double* cost,
    int_const, int_const,  # const int* kmin, const int* kmax,
    bool_const_2d, bool_const_2d,  # const bool* limitlist, const bool* optionallist
]

flags = ['ALIGNED', 'C_CONTIGUOUS']


def sociallyOptimalAssortmentVector(
    T, n, v, r, cost, allowed, mandatory, limit, a, b, verbose=False
):
    optionalList = allowed & ~mandatory
    limitList = optionalList & limit

    kmin = np.sum(limitList & mandatory, axis=1)
    kmax = np.sum(limitList & allowed, axis=1)

    initialM = np.array(mandatory, dtype=bool, order="C")
    initialM[allowed] += ~limit[allowed]

    optM = initialM
    optZ = np.zeros(T, dtype=float, order="C")

    U = cost.shape[1]

    lib.iter_vec(
        optZ, optM,  # double* optZ, pybool* optM,
        T, n, U,  # const int T, const int n, const int U,
        v, r,  # const double* v, const double* vr, const double *r,
        a, b, cost,  # const double a, const double b, const double* cost,
        kmin, kmax,  # const int kmin, const int kmax,
        limitList, optionalList,  # const pyint* limitlist, const pyint* optionallist
    )

    indices = np.arange(n)
    optM = {t: list(indices[optM[t, :]]) for t in range(T)}

    return optZ, optM
